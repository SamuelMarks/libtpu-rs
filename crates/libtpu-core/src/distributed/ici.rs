//! Deadlock-free 2D Torus (`v2`/`v3`/`v5e`/`v6e`) and 3D Twisted-Torus (`v4`/`v5p`/`v7x`)
//! optical Inter-Chip Interconnect (`ICI`) router and multi-algorithm collective engine (`Phase 8.2`).
//!
//! Implements:
//! - Generation-exact 2D Torus and 3D Twisted-Torus (`K×K×2K`, `K×2K×2K`, `K×2K×NK`) mesh geometry
//!   with Z-axis optical wrap twist `(x, y, Lz - 1) <-> ((x + twist_x) % Lx, (y + twist_y) % Ly, 0)`.
//! - Subslice boundary clamping (`SubsliceWindow`), disabling wrap-around along soft-sliced axes.
//! - Deadlock-free Dally-Seitz Virtual Channel (`VC0` / `VC1`) dateline Dimension-Order Routing (DOR)
//!   and link-health / faulty-chip BFS bypass routing (`DeadlockFreeIciRouter`), synthesizing both
//!   [`crate::proto::IciRoutingTableProto`] and [`crate::proto::SuperpodIciNetworkConfigProto`]
//!   (`superpod.routing.proto.IciNetworkConfig`).
//! - Multi-Ring, 2D/3D Mesh-Torus, and Recursive Halving-Doubling collective schedules (`IciCollectiveEngine`)
//!   across all 29 [`ElementType`]s for `AllReduce`, `ReduceScatter`, `AllGather`, `AllToAll`,
//!   `CollectivePermute`, and `RaggedAllToAll`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    DeepseaRoutingEntryProto, DeepseaRoutingTableProto, DeepseaUnicastRoutingTableProto,
    DeepseaUnicastTargetProto, DeepseaVirtualChannelControlProto, IciRouteHopProto,
    IciRoutingTableProto, SuperpodChipCoordinateProto, SuperpodChipLocationToIciConfigProto,
    SuperpodIciNetworkConfigProto, SuperpodTopologyProto, SuperpodToroidalTopologyProto,
    SuperpodTwistedTorusShapeProto, deepsea_routing_table, superpod_topology,
    superpod_toroidal_topology,
};
use crate::types::{ChipCoordinates, CollectiveReductionOp, ElementType, TpuGeneration};

/// Subslice coordinate clamping window (`[origin, origin + bounds)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsliceWindow {
    /// Inclusive lower-bound chip coordinates `(origin_x, origin_y, origin_z)`.
    pub origin: ChipCoordinates,
    /// Subslice axis extents `(len_x, len_y, len_z)`.
    pub bounds: ChipCoordinates,
}

impl SubsliceWindow {
    /// Creates a new [`SubsliceWindow`] starting at `origin` with extents `bounds`.
    ///
    /// # Arguments
    ///
    /// * `origin` - Inclusive lower-bound chip coordinates.
    /// * `bounds` - Subslice extents along `(X, Y, Z)`.
    #[must_use]
    pub const fn new(origin: ChipCoordinates, bounds: ChipCoordinates) -> Self {
        Self {
            origin,
            bounds: ChipCoordinates::new(
                if bounds.x == 0 { 1 } else { bounds.x },
                if bounds.y == 0 { 1 } else { bounds.y },
                if bounds.z == 0 { 1 } else { bounds.z },
            ),
        }
    }

    /// Returns `true` if `coord` lies strictly inside `[origin, origin + bounds)`.
    ///
    /// # Arguments
    ///
    /// * `coord` - Candidate [`ChipCoordinates`].
    #[must_use]
    pub const fn contains(&self, coord: ChipCoordinates) -> bool {
        coord.x >= self.origin.x
            && coord.x < self.origin.x.saturating_add(self.bounds.x)
            && coord.y >= self.origin.y
            && coord.y < self.origin.y.saturating_add(self.bounds.y)
            && coord.z >= self.origin.z
            && coord.z < self.origin.z.saturating_add(self.bounds.z)
    }
}

/// Single directed optical ICI hop produced by [`DeadlockFreeIciRouter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IciRoutedHop {
    /// Source chip coordinates.
    pub src: ChipCoordinates,
    /// Next-hop neighbor chip coordinates.
    pub dst: ChipCoordinates,
    /// Hardware egress link port (`0..=5` corresponding to `+X, -X, +Y, -Y, +Z, -Z`).
    pub egress_port: u32,
    /// Deadlock-free virtual channel (`0` = `VC0` pre-dateline, `1` = `VC1` post-dateline).
    pub virtual_channel: u8,
    /// Whether this hop traverses a wrap-around (or twisted wrap-around) optical link.
    pub is_wrap_around: bool,
}

/// Generation-specific 2D Torus or 3D Twisted-Torus physical ICI fabric specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwistedTorusMeshSpec {
    /// Cloud TPU hardware generation (`V2`..`V7x`).
    pub generation: TpuGeneration,
    /// Physical pod/slice mesh extents `(Lx, Ly, Lz)`.
    pub mesh_bounds: ChipCoordinates,
    /// Per-axis torus wrap-around enables `(wrap_x, wrap_y, wrap_z)`.
    pub axis_wrap: (bool, bool, bool),
    /// Canonical `superpod.routing.proto.TwistedTorusShape` classification.
    pub twisted_shape: SuperpodTwistedTorusShapeProto,
    /// Optical `(twist_x, twist_y)` coordinate shift applied when wrapping across `Z = Lz - 1 <-> 0`.
    pub z_wrap_twist: (u32, u32),
}

impl TwistedTorusMeshSpec {
    /// Constructs the hardware-accurate 2D Torus (`V2`, `V3`, `V5e`, `V6e`) or 3D Twisted-Torus
    /// (`V4`, `V5p`, `V7x`) fabric specification for `generation` and `bounds`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `bounds` - Physical mesh extents `(Lx, Ly, Lz)`.
    #[must_use]
    pub const fn from_generation_and_bounds(
        generation: TpuGeneration,
        bounds: ChipCoordinates,
    ) -> Self {
        let lx = if bounds.x == 0 { 1 } else { bounds.x };
        let ly = if bounds.y == 0 { 1 } else { bounds.y };
        let is_3d_gen = matches!(
            generation,
            TpuGeneration::V4 | TpuGeneration::V5p | TpuGeneration::V7x
        );
        let lz = if is_3d_gen {
            if bounds.z == 0 { 1 } else { bounds.z }
        } else {
            1
        };
        let norm_bounds = ChipCoordinates::new(lx, ly, lz);
        let wrap_x = lx >= 4;
        let wrap_y = ly >= 4;
        let wrap_z = is_3d_gen && lz >= 4;

        let (twisted_shape, z_wrap_twist) = if !wrap_z {
            (
                SuperpodTwistedTorusShapeProto::TwistShapeUnspecified,
                (0, 0),
            )
        } else if lx == ly && lz == 2 * lx {
            (SuperpodTwistedTorusShapeProto::TwistShapeKK2k, (lx / 2, 0))
        } else if ly == 2 * lx && lz == 2 * lx {
            (SuperpodTwistedTorusShapeProto::TwistShapeK2k2k, (0, ly / 2))
        } else if ly == 2 * lx && lz > 2 * lx && (lz % lx) == 0 {
            (
                SuperpodTwistedTorusShapeProto::TwistShapeK2kNk,
                (lx / 2, ly / 2),
            )
        } else {
            (SuperpodTwistedTorusShapeProto::TwistShapeCustomized, (1, 0))
        };

        Self {
            generation,
            mesh_bounds: norm_bounds,
            axis_wrap: (wrap_x, wrap_y, wrap_z),
            twisted_shape,
            z_wrap_twist,
        }
    }
}

/// Deadlock-free 2D Torus and 3D Twisted-Torus optical ICI router with subslice clamping and
/// link-health / vertex-fault bypass routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeadlockFreeIciRouter {
    /// Physical 2D/3D torus specification.
    pub spec: TwistedTorusMeshSpec,
    /// Optional active soft-subslice clamp window.
    pub subslice: Option<SubsliceWindow>,
    /// Directed optical links marked unhealthy `(src, dst)`.
    pub faulted_links: BTreeSet<(ChipCoordinates, ChipCoordinates)>,
    /// Chip vertices marked offline/faulty.
    pub faulted_chips: BTreeSet<ChipCoordinates>,
}

impl DeadlockFreeIciRouter {
    /// Creates a new [`DeadlockFreeIciRouter`] for `generation` and `mesh_bounds`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Cloud TPU hardware generation (`V2`..`V7x`).
    /// * `mesh_bounds` - Physical mesh extents `(Lx, Ly, Lz)`.
    #[must_use]
    pub const fn new(generation: TpuGeneration, mesh_bounds: ChipCoordinates) -> Self {
        Self {
            spec: TwistedTorusMeshSpec::from_generation_and_bounds(generation, mesh_bounds),
            subslice: None,
            faulted_links: BTreeSet::new(),
            faulted_chips: BTreeSet::new(),
        }
    }

    /// Restricts all routing strictly inside `window` and disables wrap-around along any axis
    /// whose subslice extent is smaller than `self.spec.mesh_bounds`.
    ///
    /// # Arguments
    ///
    /// * `window` - Active [`SubsliceWindow`] boundary.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IciRoutingError`] if `window` exceeds `self.spec.mesh_bounds`.
    pub fn set_subslice_window(&mut self, window: SubsliceWindow) -> LibTpuResult<()> {
        let max_x = window.origin.x.saturating_add(window.bounds.x);
        let max_y = window.origin.y.saturating_add(window.bounds.y);
        let max_z = window.origin.z.saturating_add(window.bounds.z);
        if max_x > self.spec.mesh_bounds.x
            || max_y > self.spec.mesh_bounds.y
            || max_z > self.spec.mesh_bounds.z
        {
            return Err(LibTpuError::IciRoutingError {
                src_chip: 0,
                dst_chip: 0,
                reason: format!(
                    "Subslice window {:?}+{:?} exceeds physical mesh {:?}",
                    window.origin, window.bounds, self.spec.mesh_bounds
                ),
            });
        }
        self.subslice = Some(window);
        Ok(())
    }

    /// Marks the directed ICI link `(src, dst)` as unhealthy.
    ///
    /// # Arguments
    ///
    /// * `src` - Source chip coordinates.
    /// * `dst` - Destination chip coordinates.
    pub fn mark_link_faulty(&mut self, src: ChipCoordinates, dst: ChipCoordinates) {
        self.faulted_links.insert((src, dst));
    }

    /// Marks chip `coord` as faulty/offline so bypass routes avoid transiting through `coord`.
    ///
    /// # Arguments
    ///
    /// * `coord` - Faulty chip coordinates.
    pub fn mark_chip_faulty(&mut self, coord: ChipCoordinates) {
        self.faulted_chips.insert(coord);
    }

    /// Computes the linear row-major chip ID (`z * Lx * Ly + y * Lx + x`) for `coord`.
    ///
    /// # Arguments
    ///
    /// * `coord` - Chip coordinates within `self.spec.mesh_bounds`.
    #[must_use]
    pub const fn chip_id_of(&self, coord: ChipCoordinates) -> u32 {
        let lx = self.spec.mesh_bounds.x;
        let ly = self.spec.mesh_bounds.y;
        (coord.z * lx * ly) + (coord.y * lx) + coord.x
    }

    /// Returns whether wrap-around links are active on `(X, Y, Z)` after accounting for `self.subslice`.
    #[must_use]
    pub const fn effective_axis_wrap(&self) -> (bool, bool, bool) {
        let (wx, wy, wz) = self.spec.axis_wrap;
        match self.subslice {
            Some(sub) => (
                wx && sub.bounds.x == self.spec.mesh_bounds.x,
                wy && sub.bounds.y == self.spec.mesh_bounds.y,
                wz && sub.bounds.z == self.spec.mesh_bounds.z,
            ),
            None => (wx, wy, wz),
        }
    }

    /// Enumerates all healthy outgoing 1-hop optical neighbors `(neighbor, port, is_wrap)` from `node`,
    /// applying subslice clamping and 3D twisted-torus Z-wrap coordinate shifts.
    ///
    /// # Arguments
    ///
    /// * `node` - Current [`ChipCoordinates`] vertex.
    fn outgoing_healthy_edges(&self, node: ChipCoordinates) -> Vec<(ChipCoordinates, u32, bool)> {
        let (min_c, ext_c) = self.subslice.map_or_else(
            || (ChipCoordinates::new(0, 0, 0), self.spec.mesh_bounds),
            |sub| (sub.origin, sub.bounds),
        );
        let (wx, wy, wz) = self.effective_axis_wrap();
        let mut candidates: Vec<(ChipCoordinates, u32, bool)> = Vec::with_capacity(6);

        // +X (port 0) & -X (port 1)
        if node.x + 1 < min_c.x + ext_c.x {
            candidates.push((ChipCoordinates::new(node.x + 1, node.y, node.z), 0, false));
        } else if wx {
            candidates.push((ChipCoordinates::new(min_c.x, node.y, node.z), 0, true));
        }
        if node.x > min_c.x {
            candidates.push((ChipCoordinates::new(node.x - 1, node.y, node.z), 1, false));
        } else if wx {
            candidates.push((
                ChipCoordinates::new(min_c.x + ext_c.x - 1, node.y, node.z),
                1,
                true,
            ));
        }

        // +Y (port 2) & -Y (port 3)
        if node.y + 1 < min_c.y + ext_c.y {
            candidates.push((ChipCoordinates::new(node.x, node.y + 1, node.z), 2, false));
        } else if wy {
            candidates.push((ChipCoordinates::new(node.x, min_c.y, node.z), 2, true));
        }
        if node.y > min_c.y {
            candidates.push((ChipCoordinates::new(node.x, node.y - 1, node.z), 3, false));
        } else if wy {
            candidates.push((
                ChipCoordinates::new(node.x, min_c.y + ext_c.y - 1, node.z),
                3,
                true,
            ));
        }

        // +Z (port 4) & -Z (port 5) with 3D twisted-torus shift on Z wrap
        if node.z + 1 < min_c.z + ext_c.z {
            candidates.push((ChipCoordinates::new(node.x, node.y, node.z + 1), 4, false));
        } else if wz {
            let tx = (node.x + self.spec.z_wrap_twist.0) % self.spec.mesh_bounds.x;
            let ty = (node.y + self.spec.z_wrap_twist.1) % self.spec.mesh_bounds.y;
            candidates.push((ChipCoordinates::new(tx, ty, min_c.z), 4, true));
        }
        if node.z > min_c.z {
            candidates.push((ChipCoordinates::new(node.x, node.y, node.z - 1), 5, false));
        } else if wz {
            let lx = self.spec.mesh_bounds.x;
            let ly = self.spec.mesh_bounds.y;
            let tx = (node.x + lx - (self.spec.z_wrap_twist.0 % lx)) % lx;
            let ty = (node.y + ly - (self.spec.z_wrap_twist.1 % ly)) % ly;
            candidates.push((ChipCoordinates::new(tx, ty, min_c.z + ext_c.z - 1), 5, true));
        }

        candidates
            .into_iter()
            .filter(|&(nxt, _, _)| {
                !self.faulted_chips.contains(&nxt) && !self.faulted_links.contains(&(node, nxt))
            })
            .collect()
    }

    /// Computes a deadlock-free route of [`IciRoutedHop`]s from `src` to `dst`, validating subslice
    /// clamping, bypassing faulty links/chips via Dimension-Order-prioritized BFS, and assigning
    /// Dally-Seitz dateline Virtual Channels (`VC0 -> VC1` upon crossing a wrap-around link).
    ///
    /// # Arguments
    ///
    /// * `src` - Source chip coordinates.
    /// * `dst` - Destination chip coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IciRoutingError`] if `src` or `dst` is out-of-bounds, outside the
    /// active subslice, marked faulty, or disconnected by link faults.
    pub fn compute_deadlock_free_route(
        &self,
        src: ChipCoordinates,
        dst: ChipCoordinates,
    ) -> LibTpuResult<Vec<IciRoutedHop>> {
        let full = SubsliceWindow::new(ChipCoordinates::new(0, 0, 0), self.spec.mesh_bounds);
        let active = self.subslice.unwrap_or(full);
        if !active.contains(src) || !active.contains(dst) {
            return Err(LibTpuError::IciRoutingError {
                src_chip: self.chip_id_of(src),
                dst_chip: self.chip_id_of(dst),
                reason: format!("Endpoint outside active window ({src} -> {dst})"),
            });
        }
        if self.faulted_chips.contains(&src) || self.faulted_chips.contains(&dst) {
            return Err(LibTpuError::IciRoutingError {
                src_chip: self.chip_id_of(src),
                dst_chip: self.chip_id_of(dst),
                reason: format!("Source or destination chip marked faulty ({src} -> {dst})"),
            });
        }
        if src == dst {
            return Ok(Vec::new());
        }

        let mut queue = VecDeque::new();
        let mut visited = BTreeSet::new();
        let mut parent: BTreeMap<ChipCoordinates, (ChipCoordinates, u32, bool)> = BTreeMap::new();
        queue.push_back(src);
        visited.insert(src);

        let mut found = false;
        while let Some(cur) = queue.pop_front() {
            if cur == dst {
                found = true;
                break;
            }
            for (nxt, port, is_wrap) in self.outgoing_healthy_edges(cur) {
                if visited.insert(nxt) {
                    parent.insert(nxt, (cur, port, is_wrap));
                    queue.push_back(nxt);
                }
            }
        }
        if !found {
            return Err(LibTpuError::IciRoutingError {
                src_chip: self.chip_id_of(src),
                dst_chip: self.chip_id_of(dst),
                reason: format!("No healthy ICI route between {src} and {dst}"),
            });
        }

        let mut raw_hops = Vec::new();
        let mut cursor = dst;
        while let Some(&(prev, port, is_wrap)) = parent.get(&cursor) {
            raw_hops.push((prev, cursor, port, is_wrap));
            cursor = prev;
        }
        raw_hops.reverse();

        // Dally-Seitz dateline rule: start in VC0, switch to VC1 upon crossing any wrap-around link
        let mut current_vc = 0_u8;
        let mut hops = Vec::with_capacity(raw_hops.len());
        for (h_src, h_dst, port, is_wrap) in raw_hops {
            if is_wrap {
                current_vc = 1;
            }
            hops.push(IciRoutedHop {
                src: h_src,
                dst: h_dst,
                egress_port: port,
                virtual_channel: current_vc,
                is_wrap_around: is_wrap,
            });
        }
        Ok(hops)
    }

    /// Synthesizes a complete [`IciRoutingTableProto`] across all active healthy chips in this
    /// 2D Torus or 3D Twisted-Torus slice/subslice.
    ///
    /// # Arguments
    ///
    /// * `topology_name` - Descriptive topology label (e.g., `"tpu_v7x_4x4x8_twisted"`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IciRoutingError`] if any active non-faulty chip pair is disconnected.
    pub fn synthesize_ici_routing_table_proto(
        &self,
        topology_name: &str,
    ) -> LibTpuResult<IciRoutingTableProto> {
        let coords = self.active_healthy_coordinates();
        let mut route_hops = Vec::new();
        for &src in &coords {
            for &dst in &coords {
                if src != dst {
                    let path = self.compute_deadlock_free_route(src, dst)?;
                    let first = path.first().copied().unwrap_or(IciRoutedHop {
                        src,
                        dst,
                        egress_port: 0,
                        virtual_channel: 0,
                        is_wrap_around: false,
                    });
                    let axis_delta = match first.egress_port {
                        0 | 2 | 4 => 1,
                        _ => -1,
                    };
                    route_hops.push(IciRouteHopProto {
                        src_chip_id: self.chip_id_of(src),
                        dst_chip_id: self.chip_id_of(dst),
                        egress_link_port: first.egress_port,
                        virtual_channel: u32::from(first.virtual_channel),
                        axis_delta,
                        is_wrap_around: first.is_wrap_around,
                    });
                }
            }
        }
        let dim = if self.spec.mesh_bounds.z > 1 { 3 } else { 2 };
        let (_, _, wz) = self.effective_axis_wrap();
        Ok(IciRoutingTableProto {
            topology_name: String::from(topology_name),
            torus_dimension: dim,
            is_twisted_torus: wz,
            route_hops,
            semaphore_pool_size: 128,
        })
    }

    /// Builds the per-chip [`SuperpodChipLocationToIciConfigProto`] unicast routing table for `src`.
    ///
    /// # Arguments
    ///
    /// * `src` - Source [`ChipCoordinates`].
    /// * `coords` - Active non-faulty mesh coordinates.
    fn build_chip_ici_config(
        &self,
        src: ChipCoordinates,
        coords: &[ChipCoordinates],
    ) -> LibTpuResult<SuperpodChipLocationToIciConfigProto> {
        let mut entries = Vec::with_capacity(coords.len());
        let mut unicast_targets = Vec::with_capacity(coords.len());
        for &dst in coords {
            let dst_id = self.chip_id_of(dst);
            let target = if src == dst {
                DeepseaUnicastTargetProto {
                    target: Some(0),
                    vc_control: Some(DeepseaVirtualChannelControlProto::VcNoChange as i32),
                    is_terminal: Some(true),
                    invalid: Some(false),
                }
            } else {
                let hops = self.compute_deadlock_free_route(src, dst)?;
                let first_port = hops.first().map_or(0, |h| h.egress_port);
                let vc_ctrl = if hops.first().is_some_and(|h| h.virtual_channel > 0) {
                    DeepseaVirtualChannelControlProto::VcForceHigh as i32
                } else {
                    DeepseaVirtualChannelControlProto::VcForceLow as i32
                };
                DeepseaUnicastTargetProto {
                    target: Some(first_port),
                    vc_control: Some(vc_ctrl),
                    is_terminal: Some(false),
                    invalid: Some(false),
                }
            };
            unicast_targets.push(target);
            entries.push(DeepseaRoutingEntryProto {
                unicast: Some(target),
                multicast: Vec::new(),
                table_index: Some(u64::from(dst_id)),
                invalid: Some(false),
                weighted_target: None,
            });
        }
        let rtable = DeepseaRoutingTableProto {
            entries,
            channel_merges: Vec::new(),
            table: Some(deepsea_routing_table::Table::UnicastTable(
                DeepseaUnicastRoutingTableProto {
                    unicast_targets,
                    channel_merges: Vec::new(),
                },
            )),
        };
        Ok(SuperpodChipLocationToIciConfigProto {
            chip_location: None,
            global_id: i32::try_from(self.chip_id_of(src)).ok(),
            chip_egress_routing_table: Some(rtable),
            chip_egress_next_hop_routing_table: None,
            link_next_hop_routing_tables: BTreeMap::new(),
            per_links_routing_tables: Vec::new(),
            gtc_config: None,
            single_chip_mode: Some(coords.len() == 1),
            links: vec![0, 1, 2, 3, 4, 5],
            coordinate: Some(SuperpodChipCoordinateProto {
                coordinates: vec![i64::from(src.x), i64::from(src.y), i64::from(src.z)],
            }),
        })
    }

    /// Synthesizes the `superpod.routing.proto.IciNetworkConfig` protobuf.
    ///
    /// Populates `ToroidalTopology`, `Faults`, and per-chip `accel_ssw.deepsea.proto.RoutingTable`
    /// unicast VC control tables.
    ///
    /// # Arguments
    ///
    /// * `network_id` - Unique slice network identifier string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IciRoutingError`] if any healthy chip pair is disconnected.
    pub fn synthesize_superpod_ici_network_config(
        &self,
        network_id: &str,
    ) -> LibTpuResult<SuperpodIciNetworkConfigProto> {
        let coords = self.active_healthy_coordinates();
        let (wx, wy, wz) = self.effective_axis_wrap();
        let faulty_vertices = self
            .faulted_chips
            .iter()
            .map(|c| SuperpodChipCoordinateProto {
                coordinates: vec![i64::from(c.x), i64::from(c.y), i64::from(c.z)],
            })
            .collect();
        let faulty_links = self
            .faulted_links
            .iter()
            .map(|(s, d)| superpod_toroidal_topology::IciLink {
                src: Some(SuperpodChipCoordinateProto {
                    coordinates: vec![i64::from(s.x), i64::from(s.y), i64::from(s.z)],
                }),
                dest: Some(SuperpodChipCoordinateProto {
                    coordinates: vec![i64::from(d.x), i64::from(d.y), i64::from(d.z)],
                }),
                direction: None,
            })
            .collect();

        let toroidal = SuperpodToroidalTopologyProto {
            x_dimension: Some(i64::from(self.spec.mesh_bounds.x)),
            y_dimension: Some(i64::from(self.spec.mesh_bounds.y)),
            z_dimension: Some(i64::from(self.spec.mesh_bounds.z)),
            x_wrapping: Some(wx),
            y_wrapping: Some(wy),
            z_wrapping: Some(wz),
            twisted_torus_shape: Some(self.spec.twisted_shape as i32),
            faults: Some(superpod_toroidal_topology::Faults {
                faulty_vertices,
                faulty_links,
            }),
            ..SuperpodToroidalTopologyProto::default()
        };

        let mut configs = Vec::with_capacity(coords.len());
        for &src in &coords {
            configs.push(self.build_chip_ici_config(src, &coords)?);
        }

        Ok(SuperpodIciNetworkConfigProto {
            num_machines: Some(i32::try_from((coords.len() / 4).max(1)).unwrap_or(1)),
            topology: Some(SuperpodTopologyProto {
                topology_type: Some(superpod_topology::TopologyType::ToroidalTopology(toroidal)),
            }),
            expected_links: Vec::new(),
            network_id: Some(String::from(network_id)),
            is_static_slice: Some(self.subslice.is_none()),
            wait_for_data_link_up_timeout: None,
            clear_wait_timeout: None,
            wait_for_gtc_reset_timeout: None,
            configs,
            deferred_programming: Some(false),
            compressed_configs: Vec::new(),
            chips_to_reset: Vec::new(),
            allow_softslicing: Some(true),
        })
    }

    /// Lists all active non-faulty chip coordinates within `self.subslice` (or the full mesh).
    #[must_use]
    pub fn active_healthy_coordinates(&self) -> Vec<ChipCoordinates> {
        let full = SubsliceWindow::new(ChipCoordinates::new(0, 0, 0), self.spec.mesh_bounds);
        let win = self.subslice.unwrap_or(full);
        let mut out = Vec::new();
        for z in win.origin.z..(win.origin.z + win.bounds.z) {
            for y in win.origin.y..(win.origin.y + win.bounds.y) {
                for x in win.origin.x..(win.origin.x + win.bounds.x) {
                    let c = ChipCoordinates::new(x, y, z);
                    if !self.faulted_chips.contains(&c) {
                        out.push(c);
                    }
                }
            }
        }
        out
    }
}

/// Collective communication schedule topology algorithm (`IciCollectiveAlgorithm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IciCollectiveAlgorithm {
    /// Bidirectional `N - 1`-phase ring schedule (`Scatter-Reduce` + `AllGather`).
    MultiRing,
    /// Dimension-order 2D/3D Mesh-Torus recursive phase reduction along X, then Y, then Z.
    MeshTorus2d3d,
    /// Logarithmic `ceil(log2(N))` Recursive Halving-Doubling schedule for latency-bound tensors.
    HalvingDoubling,
}

/// Single communication phase inside a synthesized [`IciCollectiveSchedule`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IciSchedulePhase {
    /// Zero-based phase index.
    pub phase_index: usize,
    /// Directed `(src_rank, dst_rank)` peer transfers executed concurrently during this phase.
    pub peer_transfers: Vec<(usize, usize)>,
}

/// Synthesized multi-phase ICI collective communication schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IciCollectiveSchedule {
    /// Selected [`IciCollectiveAlgorithm`].
    pub algorithm: IciCollectiveAlgorithm,
    /// Number of participating replicas `N`.
    pub num_participants: usize,
    /// Ordered communication phases.
    pub phases: Vec<IciSchedulePhase>,
}

/// Multi-algorithm ICI collective planner and 29-`ElementType` numerical execution engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct IciCollectiveEngine;

impl IciCollectiveEngine {
    /// Synthesizes an [`IciCollectiveSchedule`] for `num_participants` using `algorithm` and `mesh_bounds`.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - [`IciCollectiveAlgorithm`] (`MultiRing`, `MeshTorus2d3d`, or `HalvingDoubling`).
    /// * `num_participants` - Number of participating replicas (`N`).
    /// * `mesh_bounds` - 2D/3D mesh dimensions used when `algorithm == MeshTorus2d3d`.
    #[must_use]
    pub fn build_schedule(
        algorithm: IciCollectiveAlgorithm,
        num_participants: usize,
        mesh_bounds: ChipCoordinates,
    ) -> IciCollectiveSchedule {
        if num_participants <= 1 {
            return IciCollectiveSchedule {
                algorithm,
                num_participants,
                phases: Vec::new(),
            };
        }
        let mut phases = Vec::new();
        match algorithm {
            IciCollectiveAlgorithm::MultiRing => {
                for step in 0..(2 * (num_participants - 1)) {
                    let transfers = (0..num_participants)
                        .map(|r| (r, (r + 1) % num_participants))
                        .collect();
                    phases.push(IciSchedulePhase {
                        phase_index: step,
                        peer_transfers: transfers,
                    });
                }
            }
            IciCollectiveAlgorithm::MeshTorus2d3d => {
                let axes = [
                    mesh_bounds.x.max(1) as usize,
                    mesh_bounds.y.max(1) as usize,
                    mesh_bounds.z.max(1) as usize,
                ];
                let mut stride = 1_usize;
                let mut phase_idx = 0_usize;
                for extent in axes {
                    if extent > 1 {
                        let mut transfers = Vec::new();
                        for r in 0..num_participants {
                            let lane = (r / stride) % extent;
                            let next_lane = (lane + 1) % extent;
                            let dst = (r - lane * stride) + next_lane * stride;
                            if dst < num_participants {
                                transfers.push((r, dst));
                            }
                        }
                        phases.push(IciSchedulePhase {
                            phase_index: phase_idx,
                            peer_transfers: transfers,
                        });
                        phase_idx += 1;
                    }
                    stride = stride.saturating_mul(extent);
                }
            }
            IciCollectiveAlgorithm::HalvingDoubling => {
                let mut dist = 1_usize;
                let mut phase_idx = 0_usize;
                while dist < num_participants {
                    let mut transfers = Vec::new();
                    for r in 0..num_participants {
                        let partner = r ^ dist;
                        if partner < num_participants {
                            transfers.push((r, partner));
                        }
                    }
                    phases.push(IciSchedulePhase {
                        phase_index: phase_idx,
                        peer_transfers: transfers,
                    });
                    dist <<= 1;
                    phase_idx += 1;
                }
            }
        }
        IciCollectiveSchedule {
            algorithm,
            num_participants,
            phases,
        }
    }

    /// Combines two raw tensor byte buffers `lhs` and `rhs` element-by-element across all **29**
    /// [`ElementType`]s (`Invalid`, `Token`, `Tuple`, `Pred`, `S2`/`U2`, `S4`/`U4`, `S8`..`S64`,
    /// `U8`..`U64`, `C64`, `C128`, `F16`, `Bf16`, `F32`, `F64`, `F8*`, `F4E2M1Fn`, `F8E8M0Fnu`).
    ///
    /// # Arguments
    ///
    /// * `lhs` - Left packed tensor bytes.
    /// * `rhs` - Right packed tensor bytes.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `op` - [`CollectiveReductionOp`] combiner (`Sum`, `Product`, `Min`, `Max`).
    #[must_use]
    pub fn reduce_pair_typed(
        lhs: &[u8],
        rhs: &[u8],
        elem_type: ElementType,
        op: CollectiveReductionOp,
    ) -> Vec<u8> {
        match elem_type {
            ElementType::Invalid | ElementType::Token | ElementType::Tuple => lhs.to_vec(),
            ElementType::Pred => lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| {
                    let pa = a != 0;
                    let pb = b != 0;
                    let res = match op {
                        CollectiveReductionOp::Sum | CollectiveReductionOp::Max => pa || pb,
                        CollectiveReductionOp::Product | CollectiveReductionOp::Min => pa && pb,
                    };
                    u8::from(res)
                })
                .collect(),
            ElementType::U64 => {
                let mut out = Vec::with_capacity(lhs.len());
                for (ca, cb) in lhs.chunks_exact(8).zip(rhs.chunks_exact(8)) {
                    let mut ba = [0_u8; 8];
                    let mut bb = [0_u8; 8];
                    ba.copy_from_slice(ca);
                    bb.copy_from_slice(cb);
                    let va = u64::from_le_bytes(ba);
                    let vb = u64::from_le_bytes(bb);
                    let vr = match op {
                        CollectiveReductionOp::Sum => va.wrapping_add(vb),
                        CollectiveReductionOp::Product => va.wrapping_mul(vb),
                        CollectiveReductionOp::Min => va.min(vb),
                        CollectiveReductionOp::Max => va.max(vb),
                    };
                    out.extend_from_slice(&vr.to_le_bytes());
                }
                out
            }
            ElementType::S2
            | ElementType::U2
            | ElementType::S4
            | ElementType::U4
            | ElementType::S8
            | ElementType::U8
            | ElementType::S16
            | ElementType::U16
            | ElementType::S32
            | ElementType::U32
            | ElementType::S64 => {
                let va = elem_type.unpack_to_i64_vec(lhs);
                let vb = elem_type.unpack_to_i64_vec(rhs);
                let combined: Vec<i64> = va
                    .iter()
                    .zip(vb.iter())
                    .map(|(&a, &b)| match op {
                        CollectiveReductionOp::Sum => a.wrapping_add(b),
                        CollectiveReductionOp::Product => a.wrapping_mul(b),
                        CollectiveReductionOp::Min => a.min(b),
                        CollectiveReductionOp::Max => a.max(b),
                    })
                    .collect();
                elem_type.pack_from_i64_slice(&combined)
            }
            ElementType::C64 => Self::reduce_complex_f32(lhs, rhs, op),
            ElementType::C128 => Self::reduce_complex_f64(lhs, rhs, op),
            ElementType::F16
            | ElementType::F32
            | ElementType::F64
            | ElementType::Bf16
            | ElementType::F8E5M2
            | ElementType::F8E4M3Fn
            | ElementType::F8E4M3B11Fnuz
            | ElementType::F8E5M2Fnuz
            | ElementType::F8E4M3Fnuz
            | ElementType::F4E2M1Fn
            | ElementType::F8E8M0Fnu => {
                let va = elem_type.unpack_to_f64_vec(lhs);
                let vb = elem_type.unpack_to_f64_vec(rhs);
                let combined: Vec<f64> = va
                    .iter()
                    .zip(vb.iter())
                    .map(|(&a, &b)| match op {
                        CollectiveReductionOp::Sum => a + b,
                        CollectiveReductionOp::Product => a * b,
                        CollectiveReductionOp::Min => a.min(b),
                        CollectiveReductionOp::Max => a.max(b),
                    })
                    .collect();
                elem_type.pack_from_f64_slice(&combined)
            }
        }
    }

    /// Combines packed `C64` (`(f32, f32)`) complex buffers using exact complex addition/multiplication.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Left-hand packed `C64` byte slice.
    /// * `rhs` - Right-hand packed `C64` byte slice.
    /// * `op` - [`CollectiveReductionOp`] combiner.
    fn reduce_complex_f32(lhs: &[u8], rhs: &[u8], op: CollectiveReductionOp) -> Vec<u8> {
        let mut out = Vec::with_capacity(lhs.len());
        for (ca, cb) in lhs.chunks_exact(8).zip(rhs.chunks_exact(8)) {
            let mut lhs_real = [0_u8; 4];
            let mut lhs_imag = [0_u8; 4];
            let mut rhs_real = [0_u8; 4];
            let mut rhs_imag = [0_u8; 4];
            lhs_real.copy_from_slice(ca.get(0..4).unwrap_or(&[0; 4]));
            lhs_imag.copy_from_slice(ca.get(4..8).unwrap_or(&[0; 4]));
            rhs_real.copy_from_slice(cb.get(0..4).unwrap_or(&[0; 4]));
            rhs_imag.copy_from_slice(cb.get(4..8).unwrap_or(&[0; 4]));
            let (ar, ai) = (f32::from_le_bytes(lhs_real), f32::from_le_bytes(lhs_imag));
            let (br, bi) = (f32::from_le_bytes(rhs_real), f32::from_le_bytes(rhs_imag));
            let (rr, ri) = match op {
                CollectiveReductionOp::Sum => (ar + br, ai + bi),
                CollectiveReductionOp::Product => (
                    f32::mul_add(ai, -bi, ar * br),
                    f32::mul_add(ai, br, ar * bi),
                ),
                CollectiveReductionOp::Min => {
                    if (ar, ai) <= (br, bi) {
                        (ar, ai)
                    } else {
                        (br, bi)
                    }
                }
                CollectiveReductionOp::Max => {
                    if (ar, ai) >= (br, bi) {
                        (ar, ai)
                    } else {
                        (br, bi)
                    }
                }
            };
            out.extend_from_slice(&rr.to_le_bytes());
            out.extend_from_slice(&ri.to_le_bytes());
        }
        out
    }

    /// Combines packed `C128` (`(f64, f64)`) complex buffers using exact complex addition/multiplication.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Left-hand packed `C128` byte slice.
    /// * `rhs` - Right-hand packed `C128` byte slice.
    /// * `op` - [`CollectiveReductionOp`] combiner.
    fn reduce_complex_f64(lhs: &[u8], rhs: &[u8], op: CollectiveReductionOp) -> Vec<u8> {
        let mut out = Vec::with_capacity(lhs.len());
        for (ca, cb) in lhs.chunks_exact(16).zip(rhs.chunks_exact(16)) {
            let mut lhs_real = [0_u8; 8];
            let mut lhs_imag = [0_u8; 8];
            let mut rhs_real = [0_u8; 8];
            let mut rhs_imag = [0_u8; 8];
            lhs_real.copy_from_slice(ca.get(0..8).unwrap_or(&[0; 8]));
            lhs_imag.copy_from_slice(ca.get(8..16).unwrap_or(&[0; 8]));
            rhs_real.copy_from_slice(cb.get(0..8).unwrap_or(&[0; 8]));
            rhs_imag.copy_from_slice(cb.get(8..16).unwrap_or(&[0; 8]));
            let (ar, ai) = (f64::from_le_bytes(lhs_real), f64::from_le_bytes(lhs_imag));
            let (br, bi) = (f64::from_le_bytes(rhs_real), f64::from_le_bytes(rhs_imag));
            let (rr, ri) = match op {
                CollectiveReductionOp::Sum => (ar + br, ai + bi),
                CollectiveReductionOp::Product => (
                    f64::mul_add(ai, -bi, ar * br),
                    f64::mul_add(ai, br, ar * bi),
                ),
                CollectiveReductionOp::Min => {
                    if (ar, ai) <= (br, bi) {
                        (ar, ai)
                    } else {
                        (br, bi)
                    }
                }
                CollectiveReductionOp::Max => {
                    if (ar, ai) >= (br, bi) {
                        (ar, ai)
                    } else {
                        (br, bi)
                    }
                }
            };
            out.extend_from_slice(&rr.to_le_bytes());
            out.extend_from_slice(&ri.to_le_bytes());
        }
        out
    }

    /// Executes `AllReduce` across `replica_buffers` using `algorithm`, `elem_type`, and `reduction_op`,
    /// returning both the reduced per-replica buffers and the synthesized [`IciCollectiveSchedule`].
    ///
    /// # Arguments
    ///
    /// * `algorithm` - [`IciCollectiveAlgorithm`] (`MultiRing`, `MeshTorus2d3d`, or `HalvingDoubling`).
    /// * `mesh_bounds` - Physical/subslice mesh extents.
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per replica).
    /// * `elem_type` - Tensor [`ElementType`] (`0..=28`).
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    #[must_use]
    pub fn execute_all_reduce_scheduled(
        algorithm: IciCollectiveAlgorithm,
        mesh_bounds: ChipCoordinates,
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> (Vec<Vec<u8>>, IciCollectiveSchedule) {
        let sched = Self::build_schedule(algorithm, replica_buffers.len(), mesh_bounds);
        let Some((first, rest)) = replica_buffers.split_first() else {
            return (Vec::new(), sched);
        };
        let mut acc = first.clone();
        for buf in rest {
            acc = Self::reduce_pair_typed(&acc, buf, elem_type, reduction_op);
        }
        (vec![acc; replica_buffers.len()], sched)
    }

    /// Splits a packed byte buffer of `elem_type` into `num_shards` equal logical shards (handling
    /// sub-byte `S2`/`U2`/`S4`/`U4`/`F4E2M1Fn` via element unpacking as well as byte-aligned types).
    ///
    /// # Arguments
    ///
    /// * `buf` - Packed input tensor byte buffer.
    /// * `elem_type` - [`ElementType`] of elements in `buf`.
    /// * `num_shards` - Number of equal logical shards to split `buf` into.
    fn split_packed_shards(buf: &[u8], elem_type: ElementType, num_shards: usize) -> Vec<Vec<u8>> {
        let n = num_shards.max(1);
        match elem_type {
            ElementType::S2 | ElementType::U2 | ElementType::S4 | ElementType::U4 => {
                let vals = elem_type.unpack_to_i64_vec(buf);
                let chunk = (vals.len() / n).max(1);
                (0..n)
                    .map(|i| {
                        let s = (i * chunk).min(vals.len());
                        let e = (s + chunk).min(vals.len());
                        elem_type.pack_from_i64_slice(vals.get(s..e).unwrap_or(&[]))
                    })
                    .collect()
            }
            ElementType::F4E2M1Fn => {
                let vals = elem_type.unpack_to_f64_vec(buf);
                let chunk = (vals.len() / n).max(1);
                (0..n)
                    .map(|i| {
                        let s = (i * chunk).min(vals.len());
                        let e = (s + chunk).min(vals.len());
                        elem_type.pack_from_f64_slice(vals.get(s..e).unwrap_or(&[]))
                    })
                    .collect()
            }
            _ => {
                let elem_bytes = elem_type.byte_size().max(1);
                let total_elems = buf.len() / elem_bytes;
                let elems_per_shard = (total_elems / n).max(1);
                let shard_bytes = elems_per_shard * elem_bytes;
                (0..n)
                    .map(|i| {
                        let s = (i * shard_bytes).min(buf.len());
                        let e = (s + shard_bytes).min(buf.len());
                        buf.get(s..e).unwrap_or(&[]).to_vec()
                    })
                    .collect()
            }
        }
    }

    /// Concatenates packed logical shards of `elem_type` into a single contiguous packed buffer.
    ///
    /// # Arguments
    ///
    /// * `shards` - Ordered packed shard byte buffers.
    /// * `elem_type` - [`ElementType`] of elements in `shards`.
    fn concat_packed_shards(shards: &[Vec<u8>], elem_type: ElementType) -> Vec<u8> {
        match elem_type {
            ElementType::S2 | ElementType::U2 | ElementType::S4 | ElementType::U4 => {
                let mut all = Vec::new();
                for s in shards {
                    all.extend(elem_type.unpack_to_i64_vec(s));
                }
                elem_type.pack_from_i64_slice(&all)
            }
            ElementType::F4E2M1Fn => {
                let mut all = Vec::new();
                for s in shards {
                    all.extend(elem_type.unpack_to_f64_vec(s));
                }
                elem_type.pack_from_f64_slice(&all)
            }
            _ => {
                let total: usize = shards.iter().map(Vec::len).sum();
                let mut out = Vec::with_capacity(total);
                for s in shards {
                    out.extend_from_slice(s);
                }
                out
            }
        }
    }

    /// Executes `ReduceScatter` across all 29 [`ElementType`]s, reducing `replica_buffers` and
    /// scattering `1 / N` shards to each participant.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - [`IciCollectiveAlgorithm`].
    /// * `mesh_bounds` - Mesh extents.
    /// * `replica_buffers` - Input tensor byte buffers per replica.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    #[must_use]
    pub fn execute_reduce_scatter_scheduled(
        algorithm: IciCollectiveAlgorithm,
        mesh_bounds: ChipCoordinates,
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> (Vec<Vec<u8>>, IciCollectiveSchedule) {
        let (reduced, sched) = Self::execute_all_reduce_scheduled(
            algorithm,
            mesh_bounds,
            replica_buffers,
            elem_type,
            reduction_op,
        );
        let Some(first) = reduced.first() else {
            return (Vec::new(), sched);
        };
        (
            Self::split_packed_shards(first, elem_type, replica_buffers.len()),
            sched,
        )
    }

    /// Executes `AllGather` across all 29 [`ElementType`]s, concatenating all replica shards.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - [`IciCollectiveAlgorithm`].
    /// * `mesh_bounds` - Mesh extents.
    /// * `replica_buffers` - Input shard byte buffers per replica.
    /// * `elem_type` - Tensor [`ElementType`].
    #[must_use]
    pub fn execute_all_gather_scheduled(
        algorithm: IciCollectiveAlgorithm,
        mesh_bounds: ChipCoordinates,
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
    ) -> (Vec<Vec<u8>>, IciCollectiveSchedule) {
        let sched = Self::build_schedule(algorithm, replica_buffers.len(), mesh_bounds);
        if replica_buffers.is_empty() {
            return (Vec::new(), sched);
        }
        let gathered = Self::concat_packed_shards(replica_buffers, elem_type);
        (vec![gathered; replica_buffers.len()], sched)
    }

    /// Executes `AllToAll` across all 29 [`ElementType`]s, splitting each replica's buffer into `N`
    /// shards and transposing shard `(src, dst)` to replica `dst`.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Input tensor byte buffers per replica.
    /// * `elem_type` - Tensor [`ElementType`].
    #[must_use]
    pub fn execute_all_to_all_typed(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
    ) -> Vec<Vec<u8>> {
        let n = replica_buffers.len();
        if n == 0 {
            return Vec::new();
        }
        let split_per_src: Vec<Vec<Vec<u8>>> = replica_buffers
            .iter()
            .map(|b| Self::split_packed_shards(b, elem_type, n))
            .collect();
        (0..n)
            .map(|dst| {
                let incoming: Vec<Vec<u8>> = split_per_src
                    .iter()
                    .map(|src_shards| src_shards.get(dst).cloned().unwrap_or_default())
                    .collect();
                Self::concat_packed_shards(&incoming, elem_type)
            })
            .collect()
    }

    /// Executes `CollectivePermute` over [`DeadlockFreeIciRouter`], validating that every
    /// `(src_rank, dst_rank)` pair has a deadlock-free healthy ICI route before transferring bytes.
    ///
    /// # Arguments
    ///
    /// * `router` - Active [`DeadlockFreeIciRouter`].
    /// * `rank_coords` - Mapping from logical replica index to [`ChipCoordinates`].
    /// * `replica_buffers` - Input tensor byte buffers per replica.
    /// * `pairs` - `(src_replica, dst_replica)` permutation pairs.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IciRoutingError`] if any pair lacks a healthy ICI route.
    pub fn execute_collective_permute_routed(
        router: &DeadlockFreeIciRouter,
        rank_coords: &[ChipCoordinates],
        replica_buffers: &[Vec<u8>],
        pairs: &[(usize, usize)],
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let zero_len = replica_buffers.first().map_or(0, Vec::len);
        let mut out = vec![vec![0_u8; zero_len]; replica_buffers.len()];
        for &(src, dst) in pairs {
            if let (Some(&sc), Some(&dc)) = (rank_coords.get(src), rank_coords.get(dst)) {
                let _ = router.compute_deadlock_free_route(sc, dc)?;
            }
            if let (Some(sb), Some(db)) = (replica_buffers.get(src), out.get_mut(dst)) {
                db.clone_from(sb);
            }
        }
        Ok(out)
    }

    /// Executes variable-length `RaggedAllToAll` across all 29 [`ElementType`]s.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Input tensor byte buffers per replica.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `input_offsets` - Per-source element start offsets `[N][N]`.
    /// * `send_sizes` - Per-source element counts `[N][N]`.
    /// * `output_offsets` - Per-destination element start offsets `[N][N]`.
    #[must_use]
    pub fn execute_ragged_all_to_all_typed(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        input_offsets: &[Vec<usize>],
        send_sizes: &[Vec<usize>],
        output_offsets: &[Vec<usize>],
    ) -> Vec<Vec<u8>> {
        let n = replica_buffers.len();
        if n == 0 {
            return Vec::new();
        }
        if matches!(
            elem_type,
            ElementType::S2 | ElementType::U2 | ElementType::S4 | ElementType::U4
        ) {
            let unpacked: Vec<Vec<i64>> = replica_buffers
                .iter()
                .map(|b| elem_type.unpack_to_i64_vec(b))
                .collect();
            return (0..n)
                .map(|dst| {
                    let mut dst_vec = unpacked.get(dst).cloned().unwrap_or_default();
                    for (src, src_vec) in unpacked.iter().enumerate() {
                        let in_off = input_offsets
                            .get(src)
                            .and_then(|r| r.get(dst))
                            .copied()
                            .unwrap_or(0);
                        let cnt = send_sizes
                            .get(src)
                            .and_then(|r| r.get(dst))
                            .copied()
                            .unwrap_or(0);
                        let out_off = output_offsets
                            .get(dst)
                            .and_then(|r| r.get(src))
                            .copied()
                            .unwrap_or(0);
                        let safe = cnt
                            .min(src_vec.len().saturating_sub(in_off))
                            .min(dst_vec.len().saturating_sub(out_off));
                        for (d_el, s_el) in dst_vec
                            .iter_mut()
                            .skip(out_off)
                            .zip(src_vec.iter().skip(in_off))
                            .take(safe)
                        {
                            *d_el = *s_el;
                        }
                    }
                    elem_type.pack_from_i64_slice(&dst_vec)
                })
                .collect();
        }
        if elem_type == ElementType::F4E2M1Fn {
            return crate::distributed::CollectiveRuntimeEngine::execute_ragged_all_to_all(
                replica_buffers,
                elem_type,
                input_offsets,
                send_sizes,
                output_offsets,
            );
        }

        let elem_bytes = elem_type.byte_size().max(1);
        (0..n)
            .map(|dst| {
                let mut dst_buf = replica_buffers.get(dst).cloned().unwrap_or_default();
                for (src, src_buf) in replica_buffers.iter().enumerate() {
                    let in_b = input_offsets
                        .get(src)
                        .and_then(|r| r.get(dst))
                        .copied()
                        .unwrap_or(0)
                        .saturating_mul(elem_bytes);
                    let len_b = send_sizes
                        .get(src)
                        .and_then(|r| r.get(dst))
                        .copied()
                        .unwrap_or(0)
                        .saturating_mul(elem_bytes);
                    let out_b = output_offsets
                        .get(dst)
                        .and_then(|r| r.get(src))
                        .copied()
                        .unwrap_or(0)
                        .saturating_mul(elem_bytes);
                    let safe_b = len_b
                        .min(src_buf.len().saturating_sub(in_b))
                        .min(dst_buf.len().saturating_sub(out_b));
                    for (d_b, s_b) in dst_buf
                        .iter_mut()
                        .skip(out_b)
                        .zip(src_buf.iter().skip(in_b))
                        .take(safe_b)
                    {
                        *d_b = *s_b;
                    }
                }
                dst_buf
            })
            .collect()
    }
}
