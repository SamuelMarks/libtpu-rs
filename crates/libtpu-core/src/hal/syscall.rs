//! Kernel driver `ioctl` and BAR0 MMIO traits and backend implementations.
//!
//! Defines [`SyscallProvider`], [`KernelDriverOps`], [`EmulatedKernelDriverBackend`], and
//! [`PosixKernelDriverBackend`].

use super::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SYNC_FLAG_BASE, BAR0_CSR_TENSORCORE_RUN_CONTROL,
    BAR0_CSR_TENSORCORE_STATUS, BAR0_CSR_TENSORCORE_SYNC_FLAG, BAR0_CSR_THERMAL_MILLICELSIUS,
    GASKET_IOCTL_MAP_DMABUF, GASKET_IOCTL_UNMAP_BUFFER, GasketInterruptEventFd,
    GasketPageTableDmaBuf, GasketPageTableIoctl, HbmDmaRingDescriptor, MappedBarAperture,
    VfioDmaMapIoctl, VfioDmaUnmapIoctl, VfioIrqSetIoctl,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ByteSize, EventFdHandle, FileDescriptor, IoctlCommand, MmioOffset};
use bytemuck::Pod;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI32, Ordering};

/// Injectable OS system call and `PCIe` BAR MMIO interface (`open`, `close`, `ioctl`, `eventfd`, `mmio`)
/// for [`crate::hal::LinuxDeviceHal`].
pub trait SyscallProvider: Send + Sync + std::fmt::Debug {
    /// Opens a device path (e.g., `"/dev/accel0"` or `"/dev/vfio/vfio"`) and returns a file descriptor.
    ///
    /// # Arguments
    ///
    /// * `path` - Device filesystem path.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] on failure.
    fn open_device(&self, path: &str) -> LibTpuResult<i32>;

    /// Issues an `ioctl` command to an open device file descriptor.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open file descriptor.
    /// * `request` - Raw `ioctl` command number (`GASKET_IOCTL_*` or `VFIO_*`).
    /// * `payload_size` - Size in bytes of the `ioctl` payload structure.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] on failure.
    fn ioctl_device(&self, fd: i32, request: u64, payload_size: usize) -> LibTpuResult<i32>;

    /// Issues an `ioctl` command passing a mutable byte slice pointing to the `#[repr(C)]` payload.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open file descriptor.
    /// * `request` - Raw `ioctl` command number (`GASKET_IOCTL_*` or `VFIO_*`).
    /// * `payload` - Mutable byte slice backing the `#[repr(C)]` `ioctl` argument structure.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] on failure.
    fn ioctl_bytes(&self, fd: i32, request: u64, payload: &mut [u8]) -> LibTpuResult<i32> {
        self.ioctl_device(fd, request, payload.len())
    }

    /// Issues a strongly-typed `#[repr(C)]` `Pod` `ioctl` command on `fd`.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open file descriptor.
    /// * `request` - Raw `ioctl` command number (`GASKET_IOCTL_*` or `VFIO_*`).
    /// * `payload` - Mutable reference to a `#[repr(C)]` `Pod` structure.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] on failure.
    fn ioctl_typed<T: Pod>(&self, fd: i32, request: u64, payload: &mut T) -> LibTpuResult<i32>
    where
        Self: Sized,
    {
        self.ioctl_bytes(fd, request, bytemuck::bytes_of_mut(payload))
    }

    /// Closes an open device file descriptor.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open file descriptor to close.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] on failure.
    fn close_device(&self, fd: i32) -> LibTpuResult<()>;
}

/// Typed `ioctl` payload passed into [`EmulatedKernelDriverBackend::apply_typed_ioctl`].
#[derive(Debug, Clone, Copy)]
pub enum IoctlPayload<'a> {
    /// Gasket page table map/unmap descriptor.
    GasketPageTable(&'a GasketPageTableIoctl),
    /// Gasket interrupt `eventfd` binding descriptor.
    GasketInterruptEventFd(&'a GasketInterruptEventFd),
    /// Gasket `dma_buf` page table mapping descriptor.
    GasketDmaBuf(&'a GasketPageTableDmaBuf),
    /// VFIO IOMMU DMA mapping descriptor.
    VfioDmaMap(&'a VfioDmaMapIoctl),
    /// VFIO IOMMU DMA unmapping descriptor.
    VfioDmaUnmap(&'a VfioDmaUnmapIoctl),
    /// VFIO MSI-X interrupt `eventfd` binding descriptor.
    VfioIrqSet(&'a VfioIrqSetIoctl),
}

/// Cycle-accurate emulated Linux Gasket/VFIO kernel driver and `PCIe` BAR0/BAR2 MMIO CSR backend.
#[derive(Debug)]
pub struct EmulatedKernelDriverBackend {
    /// BAR0 MMIO 64-bit registers indexed by byte offset.
    bar0_registers: Mutex<BTreeMap<u64, u64>>,
    /// Hardware HBM DMA descriptor ring buffer populated by [`crate::hal::LinuxDeviceHal`].
    dma_ring: Mutex<Vec<HbmDmaRingDescriptor>>,
    /// Emulated IOMMU page table mapping IOVA addresses to mapped byte sizes.
    iommu_mappings: Mutex<BTreeMap<u64, u64>>,
    /// Registered interrupt `eventfd` handles indexed by interrupt vector ID.
    interrupt_eventfds: Mutex<BTreeMap<u32, i32>>,
    /// Total BAR0 aperture size in bytes.
    bar0_size: usize,
    /// Next allocated file descriptor.
    next_fd: AtomicI32,
}

impl Default for EmulatedKernelDriverBackend {
    fn default() -> Self {
        Self::new(0x0010_0000)
    }
}

impl EmulatedKernelDriverBackend {
    /// Creates a new [`EmulatedKernelDriverBackend`] with the given BAR0 aperture size.
    ///
    /// # Arguments
    ///
    /// * `bar0_size` - Size of the simulated `PCIe` BAR0 MMIO region in bytes.
    #[must_use]
    pub fn new(bar0_size: usize) -> Self {
        let mut regs = BTreeMap::new();
        regs.insert(BAR0_CSR_TENSORCORE_RUN_CONTROL, 0);
        regs.insert(BAR0_CSR_TENSORCORE_SYNC_FLAG, 1);
        regs.insert(BAR0_CSR_TENSORCORE_STATUS, 0);
        regs.insert(BAR0_CSR_SYNC_FLAG_BASE, 1);
        regs.insert(BAR0_CSR_HBM_DMA_RING_HEAD, 0);
        regs.insert(BAR0_CSR_HBM_DMA_RING_TAIL, 0);
        regs.insert(BAR0_CSR_THERMAL_MILLICELSIUS, 45_000);
        regs.insert(BAR0_CSR_POWER_MILLIWATTS, 175_000);
        regs.insert(BAR0_CSR_DUTY_CYCLE_PERCENT, 100);
        Self {
            bar0_registers: Mutex::new(regs),
            dma_ring: Mutex::new(Vec::new()),
            iommu_mappings: Mutex::new(BTreeMap::new()),
            interrupt_eventfds: Mutex::new(BTreeMap::new()),
            bar0_size,
            next_fd: AtomicI32::new(100),
        }
    }

    /// Appends a 32-byte [`HbmDmaRingDescriptor`] to the emulated HBM DMA descriptor ring.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - Hardware DMA ring descriptor to enqueue.
    pub fn enqueue_dma_descriptor(&self, descriptor: HbmDmaRingDescriptor) {
        self.dma_ring.lock().push(descriptor);
    }

    /// Returns the number of descriptors submitted to the HBM DMA descriptor ring.
    #[must_use]
    pub fn dma_ring_len(&self) -> usize {
        self.dma_ring.lock().len()
    }

    /// Applies a typed [`IoctlPayload`] to the emulated IOMMU page table or interrupt vector table.
    ///
    /// # Arguments
    ///
    /// * `request` - Raw `ioctl` command number.
    /// * `payload` - Typed [`IoctlPayload`] structure.
    pub fn apply_typed_ioctl(&self, request: u64, payload: IoctlPayload<'_>) {
        match payload {
            IoctlPayload::GasketPageTable(pt) => {
                let mut guard = self.iommu_mappings.lock();
                if request == GASKET_IOCTL_UNMAP_BUFFER {
                    guard.remove(&pt.device_address);
                } else {
                    guard.insert(pt.device_address, pt.size);
                }
            }
            IoctlPayload::GasketDmaBuf(db) => {
                self.iommu_mappings.lock().insert(db.device_address, 4096);
            }
            IoctlPayload::VfioDmaMap(vm) => {
                self.iommu_mappings.lock().insert(vm.iova, vm.size);
            }
            IoctlPayload::VfioDmaUnmap(vu) => {
                self.iommu_mappings.lock().remove(&vu.iova);
            }
            IoctlPayload::GasketInterruptEventFd(ge) => {
                self.interrupt_eventfds.lock().insert(
                    u32::try_from(ge.interrupt).unwrap_or(0),
                    i32::try_from(ge.event_fd).unwrap_or(0),
                );
            }
            IoctlPayload::VfioIrqSet(vi) => {
                self.interrupt_eventfds.lock().insert(vi.index, vi.event_fd);
            }
        }
    }

    /// Returns the number of active IOVA mappings in the emulated IOMMU page table.
    pub fn mapped_iova_count(&self) -> usize {
        self.iommu_mappings.lock().len()
    }

    /// Returns the registered `eventfd` for interrupt `index`, if present.
    ///
    /// # Arguments
    ///
    /// * `index` - Interrupt vector index.
    pub fn registered_interrupt_fd(&self, index: u32) -> Option<i32> {
        self.interrupt_eventfds.lock().get(&index).copied()
    }

    /// Clears all registered interrupt `eventfd` bindings.
    pub fn clear_interrupt_fds(&self) {
        self.interrupt_eventfds.lock().clear();
    }

    /// Creates a simulated Linux `eventfd` handle.
    pub fn create_eventfd(&self) -> EventFdHandle {
        let efd = self.next_fd.fetch_add(1, Ordering::Relaxed);
        EventFdHandle(efd)
    }

    /// Reads a 64-bit BAR0 CSR value directly from the register map.
    ///
    /// # Arguments
    ///
    /// * `csr_offset` - Byte offset of the BAR0 CSR.
    pub fn get_csr(&self, csr_offset: u64) -> u64 {
        self.bar0_registers
            .lock()
            .get(&csr_offset)
            .copied()
            .unwrap_or(0)
    }

    /// Writes a 64-bit BAR0 CSR value directly into the register map.
    ///
    /// # Arguments
    ///
    /// * `csr_offset` - Byte offset of the BAR0 CSR.
    /// * `value` - 64-bit register value to store.
    pub fn set_csr(&self, csr_offset: u64, value: u64) {
        self.bar0_registers.lock().insert(csr_offset, value);
    }

    /// Reads a 64-bit CSR from `PCIe` BAR `bar_index` at `offset`.
    ///
    /// # Arguments
    ///
    /// * `bar_index` - `PCIe` BAR index (`0` or `2`).
    /// * `offset` - Byte offset within the BAR.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset + 8 > bar0_size`.
    pub fn read_mmio_u64(&self, bar_index: u8, offset: MmioOffset) -> LibTpuResult<u64> {
        if bar_index != 0
            || usize::try_from(offset.0)
                .unwrap_or(usize::MAX)
                .saturating_add(8)
                > self.bar0_size
        {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index,
                offset: offset.0,
                size: 8,
            });
        }
        let val = self
            .bar0_registers
            .lock()
            .get(&offset.0)
            .copied()
            .unwrap_or(0);
        Ok(val)
    }

    /// Writes a 64-bit CSR to `PCIe` BAR `bar_index` at `offset`.
    ///
    /// # Arguments
    ///
    /// * `bar_index` - `PCIe` BAR index (`0` or `2`).
    /// * `offset` - Byte offset within the BAR.
    /// * `value` - 64-bit register value to write.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset + 8 > bar0_size`.
    pub fn write_mmio_u64(
        &self,
        bar_index: u8,
        offset: MmioOffset,
        value: u64,
    ) -> LibTpuResult<()> {
        if bar_index != 0
            || usize::try_from(offset.0)
                .unwrap_or(usize::MAX)
                .saturating_add(8)
                > self.bar0_size
        {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index,
                offset: offset.0,
                size: 8,
            });
        }
        let mut guard = self.bar0_registers.lock();
        guard.insert(offset.0, value);
        if offset.0 == BAR0_CSR_TENSORCORE_RUN_CONTROL && value != 0 {
            let sync = guard
                .get(&BAR0_CSR_TENSORCORE_SYNC_FLAG)
                .copied()
                .unwrap_or(0);
            guard.insert(BAR0_CSR_TENSORCORE_SYNC_FLAG, sync.wrapping_add(1));
        }
        drop(guard);
        Ok(())
    }

    /// Maps a `dma_buf` file descriptor into the Gasket page table (`GASKET_IOCTL_MAP_DMABUF`).
    ///
    /// # Arguments
    ///
    /// * `fd` - Open Gasket device file descriptor.
    /// * `payload` - `GasketPageTableDmaBuf` descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] if the `ioctl` fails.
    pub fn map_dmabuf(
        &self,
        fd: FileDescriptor,
        payload: &GasketPageTableDmaBuf,
    ) -> LibTpuResult<()> {
        self.ioctl_device(fd.0, GASKET_IOCTL_MAP_DMABUF, size_of_val(payload))?;
        self.apply_typed_ioctl(GASKET_IOCTL_MAP_DMABUF, IoctlPayload::GasketDmaBuf(payload));
        Ok(())
    }
}

impl SyscallProvider for EmulatedKernelDriverBackend {
    fn open_device(&self, _path: &str) -> LibTpuResult<i32> {
        let fd = self.next_fd.fetch_add(1, Ordering::Relaxed);
        Ok(fd)
    }

    fn ioctl_device(&self, fd: i32, request: u64, _payload_size: usize) -> LibTpuResult<i32> {
        let _ = IoctlCommand(request);
        if fd < 0 {
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: 9,
                message: String::from("EBADF"),
            });
        }
        Ok(0)
    }

    fn close_device(&self, fd: i32) -> LibTpuResult<()> {
        if fd < 0 {
            return Err(LibTpuError::HardwareIoError {
                operation: "close",
                errno: 9,
                message: String::from("EBADF"),
            });
        }
        Ok(())
    }
}

/// Alias for [`EmulatedKernelDriverBackend`] (`SimulatedKernelDriverBackend`).
pub type SimulatedKernelDriverBackend = EmulatedKernelDriverBackend;

/// Virtual `PCIe`/VFIO/Gasket kernel loopback device backend (`VirtualTpuKernelDevice`).
pub type VirtualTpuKernelDevice = EmulatedKernelDriverBackend;

/// Mock kernel syscall provider for deterministic cross-platform unit testing (`MockKernelSyscalls`).
pub type MockKernelSyscalls = EmulatedKernelDriverBackend;

/// Real Linux/POSIX syscall provider (`RealLinuxSyscalls`).
pub type RealLinuxSyscalls = PosixKernelDriverBackend;

/// Injectable kernel driver operations trait (`KernelDriverOps`) extending [`SyscallProvider`].
pub trait KernelDriverOps: SyscallProvider {}
impl<T: SyscallProvider + ?Sized> KernelDriverOps for T {}

/// Generic VFS `FIOCLEX` `ioctl` command (`0x20006601` on macOS, `0x5451` on Linux).
#[cfg(target_os = "macos")]
pub const POSIX_FIOCLEX_IOCTL: u64 = 0x2000_6601;

/// Generic VFS `FIOCLEX` `ioctl` command (`0x5451` on Linux).
#[cfg(not(target_os = "macos"))]
pub const POSIX_FIOCLEX_IOCTL: u64 = 0x5451;

/// Low-level POSIX C library (`libc`) FFI wrappers for [`PosixKernelDriverBackend`].
mod posix_libc {
    pub(super) use libc::{
        MAP_ANON as MAP_ANONYMOUS, MAP_PRIVATE, MAP_SHARED, O_RDWR, PROT_READ, PROT_WRITE, close,
        ioctl, mmap, munmap, open, pipe,
    };
}

/// Real POSIX kernel driver backend invoking `libc` `open`, `close`, `ioctl`, `mmap`, `munmap`,
/// and `pipe`/`eventfd` system calls for `/dev/accel*` and `/dev/vfio/*` device nodes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PosixKernelDriverBackend;

impl PosixKernelDriverBackend {
    /// Creates a new [`PosixKernelDriverBackend`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Issues a typed `#[repr(C)]` `ioctl` command on `fd` passing a mutable pointer to `payload`.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open POSIX file descriptor.
    /// * `request` - Raw `ioctl` command number.
    /// * `payload` - Mutable reference to a `#[repr(C)]` `ioctl` payload structure.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] if `libc::ioctl` returns `< 0`.
    pub fn ioctl_with_payload<T>(
        &self,
        fd: i32,
        request: u64,
        payload: &mut T,
    ) -> LibTpuResult<i32> {
        #[allow(clippy::cast_possible_truncation)]
        let req_ulong = request as libc::c_ulong;
        let ptr = std::ptr::from_mut::<T>(payload).cast::<libc::c_void>();
        // SAFETY: `ptr` points to a valid, mutable `T` structure on the stack.
        let rc = unsafe { posix_libc::ioctl(fd, req_ulong, ptr) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: err.raw_os_error().unwrap_or(9),
                message: err.to_string(),
            });
        }
        Ok(rc)
    }

    /// Maps a `PCIe` BAR aperture of `size` bytes into a [`MappedBarAperture`].
    ///
    /// # Arguments
    ///
    /// * `fd` - Open device file descriptor (or `-1` for anonymous mapping).
    /// * `size` - Mapping byte size (must be `> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `mmap` fails.
    pub fn mmap_bar_aperture(
        &self,
        fd: FileDescriptor,
        size: ByteSize,
    ) -> LibTpuResult<MappedBarAperture> {
        let base_vaddr = self.mmap_bar0(fd.0, size.0)?;
        Ok(MappedBarAperture { base_vaddr, size })
    }

    /// Unmaps a [`MappedBarAperture`] previously mapped with [`PosixKernelDriverBackend::mmap_bar_aperture`].
    ///
    /// # Arguments
    ///
    /// * `aperture` - Mapped BAR aperture to unmap.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `munmap` fails.
    pub fn munmap_bar_aperture(&self, aperture: MappedBarAperture) -> LibTpuResult<()> {
        self.munmap_bar0(aperture.base_vaddr, aperture.size.0)
    }

    /// Maps a `PCIe` BAR0 region of `size` bytes on file descriptor `fd` (or anonymous memory when `fd == -1`).
    ///
    /// # Arguments
    ///
    /// * `fd` - Open device file descriptor (or `-1` for anonymous mapping).
    /// * `size` - Mapping byte size (must be `> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `mmap` returns `MAP_FAILED`.
    pub fn mmap_bar0(&self, fd: i32, size: usize) -> LibTpuResult<usize> {
        let flags = if fd < 0 {
            posix_libc::MAP_PRIVATE | posix_libc::MAP_ANONYMOUS
        } else {
            posix_libc::MAP_SHARED
        };
        // SAFETY: `mmap` with `null_mut()` lets the kernel select a page-aligned virtual address.
        let ptr = unsafe {
            posix_libc::mmap(
                std::ptr::null_mut(),
                size,
                posix_libc::PROT_READ | posix_libc::PROT_WRITE,
                flags,
                fd,
                0,
            )
        };
        if ptr as usize == usize::MAX {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "mmap",
                errno: err.raw_os_error().unwrap_or(22),
                message: err.to_string(),
            });
        }
        Ok(ptr as usize)
    }

    /// Unmaps a `PCIe` BAR0 region previously mapped with [`PosixKernelDriverBackend::mmap_bar0`].
    ///
    /// # Arguments
    ///
    /// * `addr` - Mapped virtual address returned by `mmap_bar0`.
    /// * `size` - Mapping byte size (must be `> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `munmap` fails.
    pub fn munmap_bar0(&self, addr: usize, size: usize) -> LibTpuResult<()> {
        // SAFETY: `munmap` validates `addr` and `size` in the kernel (`size == 0` returns `EINVAL`).
        let rc = unsafe { posix_libc::munmap(addr as *mut std::ffi::c_void, size) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "munmap",
                errno: err.raw_os_error().unwrap_or(22),
                message: err.to_string(),
            });
        }
        Ok(())
    }

    /// Allocates an anonymous read-write memory-mapped `PCIe` BAR0 buffer of `size` bytes
    /// using `memmap2::MmapOptions`.
    ///
    /// # Arguments
    ///
    /// * `size` - Mapping byte size (must be `> 0` and within OS virtual address limits).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `size == 0` or `memmap2` allocation fails.
    pub fn allocate_anon_bar0_mmap(&self, size: ByteSize) -> LibTpuResult<memmap2::MmapMut> {
        let req_len = if size.0 == 0 { usize::MAX } else { size.0 };
        memmap2::MmapOptions::new()
            .len(req_len)
            .map_anon()
            .map_err(|err| LibTpuError::HardwareIoError {
                operation: "memmap2::map_anon",
                errno: err
                    .raw_os_error()
                    .unwrap_or(nix::errno::Errno::EINVAL as i32),
                message: err.to_string(),
            })
    }

    /// Allocates an OS-backed event notification descriptor (`eventfd` / pipe read end).
    #[must_use]
    pub fn create_eventfd(&self) -> EventFdHandle {
        let mut fds = [0_i32; 2];
        // SAFETY: `fds` points to a valid 2-element `c_int` array.
        let _ = unsafe { posix_libc::pipe(fds.as_mut_ptr()) };
        // SAFETY: Closing the write end of the newly created pipe.
        let _ = unsafe { posix_libc::close(fds[1]) };
        EventFdHandle(fds[0])
    }
}

impl SyscallProvider for PosixKernelDriverBackend {
    fn open_device(&self, path: &str) -> LibTpuResult<i32> {
        let c_path = std::ffi::CString::new(path).map_err(|_| LibTpuError::InvalidArgument {
            parameter: "path",
            reason: String::from("Device path contains interior NUL byte"),
        })?;
        // SAFETY: `c_path.as_ptr()` is a valid NUL-terminated C string.
        let fd = unsafe { posix_libc::open(c_path.as_ptr(), posix_libc::O_RDWR) };
        if fd < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "open",
                errno: err.raw_os_error().unwrap_or(2),
                message: err.to_string(),
            });
        }
        Ok(fd)
    }

    fn ioctl_device(&self, fd: i32, request: u64, payload_size: usize) -> LibTpuResult<i32> {
        let mut arg_buf = vec![0_u8; payload_size.max(64)];
        self.ioctl_bytes(fd, request, &mut arg_buf)
    }

    fn ioctl_bytes(&self, fd: i32, request: u64, payload: &mut [u8]) -> LibTpuResult<i32> {
        #[allow(clippy::cast_possible_truncation)]
        let req_ulong = request as std::ffi::c_ulong;
        // SAFETY: `payload` is a valid mutable byte slice whose pointer is passed to the kernel.
        let rc = unsafe {
            posix_libc::ioctl(
                fd,
                req_ulong,
                payload.as_mut_ptr().cast::<std::ffi::c_void>(),
            )
        };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: err.raw_os_error().unwrap_or(9),
                message: err.to_string(),
            });
        }
        Ok(rc)
    }

    fn close_device(&self, fd: i32) -> LibTpuResult<()> {
        // SAFETY: `close` is safe for any integer file descriptor (returns `-1` with `EBADF` if invalid).
        let rc = unsafe { posix_libc::close(fd) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "close",
                errno: err.raw_os_error().unwrap_or(9),
                message: err.to_string(),
            });
        }
        Ok(())
    }
}
