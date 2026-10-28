/**
 * @file libtpu.cpp
 * @brief Implementation of the strongly-typed C++20 RAII wrapper classes for
 * `libtpu-rs` driving `GetPjrtApi()`, `GetLibtpuSdkApi()`, `TpuTopology_*()`,
 * and `TpuProfiler_*()` directly via typed `PJRT_*_Args` and
 * `LibtpuSdk_*_Args` C structures.
 */

#include "libtpu.hpp"

namespace libtpu {

static_assert(sizeof(PJRT_Api) == 1144, "PJRT_Api must be 1144 bytes");
static_assert(sizeof(LibtpuSdk_Api) == 144, "LibtpuSdk_Api must be 144 bytes");
static_assert(sizeof(PJRT_Client_Create_Args) == 88,
              "PJRT_Client_Create_Args must be 88 bytes");
static_assert(sizeof(PJRT_Client_BufferFromHostBuffer_Args) == 120,
              "PJRT_Client_BufferFromHostBuffer_Args must be 120 bytes");
static_assert(sizeof(PJRT_LoadedExecutable_Execute_Args) == 80,
              "PJRT_LoadedExecutable_Execute_Args must be 80 bytes");
static_assert(sizeof(PJRT_Buffer_ToHostBuffer_Args) == 56,
              "PJRT_Buffer_ToHostBuffer_Args must be 56 bytes");

/**
 * @brief Converts a `PJRT_Error*` returned by a `GetPjrtApi()` slot into a
 * `Status` (querying both `PJRT_Error_GetCode` and `PJRT_Error_Message`) and
 * frees `err`.
 * @param err Pointer to `PJRT_Error` (or `nullptr` on success).
 * @return `Status` representing success (`0`) or the queried `PJRT_Error_Code`
 * and message.
 */
static Status pjrt_error_to_status(PJRT_Error *err) noexcept {
  if (err == nullptr) {
    return Status(LibTpuCStatus{0, nullptr}, "");
  }
  PJRT_Error_GetCode_Args code_args{sizeof(PJRT_Error_GetCode_Args), nullptr,
                                    err, 0, 0};
  (void)GetPjrtApi()->pjrt_error_get_code(&code_args);

  PJRT_Error_Message_Args msg_args{sizeof(PJRT_Error_Message_Args), nullptr,
                                   err, nullptr, 0};
  GetPjrtApi()->pjrt_error_message(&msg_args);

  std::string owned_msg(msg_args.message, msg_args.message_size);

  PJRT_Error_Destroy_Args del_args{sizeof(PJRT_Error_Destroy_Args), nullptr,
                                   err};
  GetPjrtApi()->pjrt_error_destroy(&del_args);
  return Status(LibTpuCStatus{code_args.code, nullptr},
                static_cast<std::string &&>(owned_msg));
}

Status::Status(LibTpuCStatus raw, std::string msg_str) noexcept
    : raw_(raw), msg_str_(static_cast<std::string &&>(msg_str)) {}

Status::~Status() noexcept { libtpu_rs_status_free(raw_); }

Status::Status(Status &&other) noexcept
    : raw_(other.raw_), msg_str_(static_cast<std::string &&>(other.msg_str_)) {
  other.raw_.code = 0;
  other.raw_.message = nullptr;
}

Status &Status::operator=(Status &&other) noexcept {
  LibTpuCStatus next = other.raw_;
  std::string next_msg = static_cast<std::string &&>(other.msg_str_);
  other.raw_.code = 0;
  other.raw_.message = nullptr;
  libtpu_rs_status_free(raw_);
  raw_ = next;
  msg_str_ = static_cast<std::string &&>(next_msg);
  return *this;
}

bool Status::is_ok() const noexcept { return raw_.code == 0; }

int32_t Status::code() const noexcept { return raw_.code; }

const char *Status::message() const noexcept { return msg_str_.c_str(); }

Buffer::Buffer(PJRT_Buffer *handle) noexcept : handle_(handle) {}

Buffer::~Buffer() noexcept {
  PJRT_Buffer_Destroy_Args args{sizeof(PJRT_Buffer_Destroy_Args), nullptr,
                                handle_};
  (void)GetPjrtApi()->pjrt_buffer_destroy(&args);
}

Buffer::Buffer(Buffer &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  PJRT_Buffer *next = other.handle_;
  other.handle_ = nullptr;
  PJRT_Buffer_Destroy_Args args{sizeof(PJRT_Buffer_Destroy_Args), nullptr,
                                handle_};
  (void)GetPjrtApi()->pjrt_buffer_destroy(&args);
  handle_ = next;
  return *this;
}

Status Buffer::to_host(uint8_t *host_dst, size_t len) const noexcept {
  PJRT_Buffer_ToHostBuffer_Args args{sizeof(PJRT_Buffer_ToHostBuffer_Args),
                                     nullptr,
                                     handle_,
                                     nullptr,
                                     host_dst,
                                     len,
                                     nullptr};
  return pjrt_error_to_status(GetPjrtApi()->pjrt_buffer_to_host_buffer(&args));
}

size_t Buffer::rank() const noexcept {
  PJRT_Buffer_Dimensions_Args args{sizeof(PJRT_Buffer_Dimensions_Args), nullptr,
                                   handle_, nullptr, 0};
  (void)GetPjrtApi()->pjrt_buffer_dimensions(&args);
  return args.num_dims;
}

int64_t Buffer::dimension(size_t axis) const noexcept {
  PJRT_Buffer_Dimensions_Args args{sizeof(PJRT_Buffer_Dimensions_Args), nullptr,
                                   handle_, nullptr, 0};
  (void)GetPjrtApi()->pjrt_buffer_dimensions(&args);
  return args.dims[axis];
}

DataType Buffer::element_type() const noexcept {
  PJRT_Buffer_ElementType_Args args{sizeof(PJRT_Buffer_ElementType_Args),
                                    nullptr, handle_, 0, 0};
  (void)GetPjrtApi()->pjrt_buffer_element_type(&args);
  return static_cast<DataType>(args.type);
}

size_t Buffer::tile_count() const noexcept {
  PJRT_Buffer_GetMemoryLayout_Args args{
      sizeof(PJRT_Buffer_GetMemoryLayout_Args),
      nullptr,
      handle_,
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  (void)GetPjrtApi()->pjrt_buffer_get_memory_layout(&args);
  return args.layout_words[6];
}

Status Buffer::copy_to_device(PJRT_Client *client, uint32_t dst_device_id,
                              Buffer *out_buffer) const noexcept {
  PJRT_Client_LookupDevice_Args look_args{
      sizeof(PJRT_Client_LookupDevice_Args), nullptr, client,
      static_cast<int32_t>(dst_device_id),   0,       nullptr};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_lookup_device(&look_args);
  if (err == nullptr) {
    PJRT_Buffer_CopyToDevice_Args cp_args{sizeof(PJRT_Buffer_CopyToDevice_Args),
                                          nullptr, handle_, look_args.device,
                                          nullptr};
    (void)GetPjrtApi()->pjrt_buffer_copy_to_device(&cp_args);
    *out_buffer = Buffer(cp_args.dst_buffer);
  }
  return pjrt_error_to_status(err);
}

const PJRT_Buffer *Buffer::raw() const noexcept { return handle_; }

LoadedExecutable::LoadedExecutable(PJRT_LoadedExecutable *handle) noexcept
    : handle_(handle) {}

LoadedExecutable::~LoadedExecutable() noexcept {
  PJRT_LoadedExecutable_Destroy_Args args{
      sizeof(PJRT_LoadedExecutable_Destroy_Args), nullptr, handle_};
  (void)GetPjrtApi()->pjrt_loaded_executable_destroy(&args);
}

LoadedExecutable::LoadedExecutable(LoadedExecutable &&other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

LoadedExecutable &
LoadedExecutable::operator=(LoadedExecutable &&other) noexcept {
  PJRT_LoadedExecutable *next = other.handle_;
  other.handle_ = nullptr;
  PJRT_LoadedExecutable_Destroy_Args args{
      sizeof(PJRT_LoadedExecutable_Destroy_Args), nullptr, handle_};
  (void)GetPjrtApi()->pjrt_loaded_executable_destroy(&args);
  handle_ = next;
  return *this;
}

Status LoadedExecutable::execute(uint32_t device_id,
                                 const PJRT_Buffer *const *inputs,
                                 size_t num_inputs, PJRT_Buffer **outputs_out,
                                 size_t num_outputs) const noexcept {
  (void)num_outputs;
  const PJRT_Buffer *const *arg_lists[1] = {inputs};
  PJRT_Buffer **out_lists[1] = {outputs_out};
  PJRT_LoadedExecutable_Execute_Args ex_args{
      sizeof(PJRT_LoadedExecutable_Execute_Args),
      nullptr,
      handle_,
      nullptr,
      arg_lists,
      static_cast<size_t>(device_id) + 1U,
      num_inputs,
      out_lists,
      nullptr,
      nullptr};
  return pjrt_error_to_status(
      GetPjrtApi()->pjrt_loaded_executable_execute(&ex_args));
}

Status LoadedExecutable::execute_binary(uint32_t device_id,
                                        const Buffer &input0,
                                        const Buffer &input1,
                                        Buffer *output_out) const noexcept {
  const PJRT_Buffer *inputs[2] = {input0.raw(), input1.raw()};
  PJRT_Buffer *out_raw = nullptr;
  Status st = execute(device_id, inputs, 2, &out_raw, 1);
  if (st.is_ok()) {
    *output_out = Buffer(out_raw);
  }
  return st;
}

Status LoadedExecutable::serialize(uint8_t *out_bytes, size_t max_len,
                                   size_t *out_len) const noexcept {
  (void)max_len;
  PJRT_LoadedExecutable_GetExecutable_Args get_ex_args{
      sizeof(PJRT_LoadedExecutable_GetExecutable_Args), nullptr, handle_,
      nullptr};
  (void)GetPjrtApi()->pjrt_loaded_executable_get_executable(&get_ex_args);
  PJRT_Executable_Serialize_Args ser_args{
      sizeof(PJRT_Executable_Serialize_Args),
      nullptr,
      get_ex_args.executable,
      nullptr,
      0,
      nullptr,
      nullptr};
  (void)GetPjrtApi()->pjrt_executable_serialize(&ser_args);
  const uint8_t *src =
      reinterpret_cast<const uint8_t *>(ser_args.serialized_bytes);
  size_t len = ser_args.serialized_bytes_size;
  *out_len = len;
  for (size_t i = 0; i < len; ++i) {
    out_bytes[i] = src[i];
  }
  return Status(LibTpuCStatus{0, nullptr}, "");
}

size_t LoadedExecutable::fingerprint(char *out_buf,
                                     size_t buf_len) const noexcept {
  (void)buf_len;
  PJRT_LoadedExecutable_Fingerprint_Args fp_args{
      sizeof(PJRT_LoadedExecutable_Fingerprint_Args), nullptr, handle_, nullptr,
      0};
  (void)GetPjrtApi()->pjrt_loaded_executable_fingerprint(&fp_args);
  const char *src = fp_args.executable_fingerprint;
  size_t len = fp_args.executable_fingerprint_size;
  for (size_t i = 0; i < len; ++i) {
    out_buf[i] = src[i];
  }
  return len;
}

float LoadedExecutable::flops() const noexcept {
  PJRT_LoadedExecutable_GetExecutable_Args get_ex_args{
      sizeof(PJRT_LoadedExecutable_GetExecutable_Args), nullptr, handle_,
      nullptr};
  (void)GetPjrtApi()->pjrt_loaded_executable_get_executable(&get_ex_args);
  PJRT_Executable_GetCostAnalysis_Args cost_args{
      sizeof(PJRT_Executable_GetCostAnalysis_Args), nullptr,
      get_ex_args.executable, nullptr, 0};
  (void)GetPjrtApi()->pjrt_executable_get_cost_analysis(&cost_args);
  return static_cast<float>(cost_args.num_properties);
}

size_t LoadedExecutable::num_outputs() const noexcept {
  PJRT_LoadedExecutable_GetExecutable_Args get_ex_args{
      sizeof(PJRT_LoadedExecutable_GetExecutable_Args), nullptr, handle_,
      nullptr};
  (void)GetPjrtApi()->pjrt_loaded_executable_get_executable(&get_ex_args);
  PJRT_Executable_OutputMemoryKinds_Args omk_args{
      sizeof(PJRT_Executable_OutputMemoryKinds_Args),
      nullptr,
      get_ex_args.executable,
      0,
      nullptr,
      nullptr};
  (void)GetPjrtApi()->pjrt_executable_output_memory_kinds(&omk_args);
  return omk_args.num_outputs;
}

Status LoadedExecutable::optimized_program(uint8_t *out_bytes, size_t max_len,
                                           size_t *out_len) const noexcept {
  PJRT_LoadedExecutable_GetExecutable_Args get_ex_args{
      sizeof(PJRT_LoadedExecutable_GetExecutable_Args), nullptr, handle_,
      nullptr};
  (void)GetPjrtApi()->pjrt_loaded_executable_get_executable(&get_ex_args);
  PJRT_Program prog{sizeof(PJRT_Program), nullptr, nullptr, 0, nullptr, 0};
  PJRT_Executable_OptimizedProgram_Args opt_args{
      sizeof(PJRT_Executable_OptimizedProgram_Args), nullptr,
      get_ex_args.executable, &prog};
  (void)GetPjrtApi()->pjrt_executable_optimized_program(&opt_args);
  *out_len = prog.code_size;
  prog.code = reinterpret_cast<const char *>(out_bytes);
  prog.code_size = max_len;
  return pjrt_error_to_status(
      GetPjrtApi()->pjrt_executable_optimized_program(&opt_args));
}

const PJRT_LoadedExecutable *LoadedExecutable::raw() const noexcept {
  return handle_;
}

Event::Event(PJRT_Event *handle) noexcept : handle_(handle) {}

Event::~Event() noexcept {
  PJRT_Event_Destroy_Args args{sizeof(PJRT_Event_Destroy_Args), nullptr,
                               handle_};
  (void)GetPjrtApi()->pjrt_event_destroy(&args);
}

Event::Event(Event &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

Event &Event::operator=(Event &&other) noexcept {
  PJRT_Event *next = other.handle_;
  other.handle_ = nullptr;
  PJRT_Event_Destroy_Args args{sizeof(PJRT_Event_Destroy_Args), nullptr,
                               handle_};
  (void)GetPjrtApi()->pjrt_event_destroy(&args);
  handle_ = next;
  return *this;
}

Status Event::create(Event *event_out) noexcept {
  PJRT_Event_Create_Args args{sizeof(PJRT_Event_Create_Args), nullptr, nullptr};
  PJRT_Error *err = GetPjrtApi()->pjrt_event_create(&args);
  *event_out = Event(args.event);
  return pjrt_error_to_status(err);
}

bool Event::is_ready() const noexcept {
  PJRT_Event_IsReady_Args args{sizeof(PJRT_Event_IsReady_Args),
                               nullptr,
                               handle_,
                               false,
                               {0, 0, 0, 0, 0, 0, 0}};
  (void)GetPjrtApi()->pjrt_event_is_ready(&args);
  return args.is_ready;
}

Status Event::set(int32_t error_code, const char *message) noexcept {
  PJRT_Event_Set_Args args{
      sizeof(PJRT_Event_Set_Args), nullptr, handle_, error_code, 0, message, 0};
  return pjrt_error_to_status(GetPjrtApi()->pjrt_event_set(&args));
}

Status Event::on_ready(PJRT_Event_OnReadyCallback callback,
                       void *user_arg) noexcept {
  PJRT_Event_OnReady_Args args{sizeof(PJRT_Event_OnReady_Args), nullptr,
                               handle_, callback, user_arg};
  return pjrt_error_to_status(GetPjrtApi()->pjrt_event_on_ready(&args));
}

Status Event::await() const noexcept {
  PJRT_Event_Await_Args args{sizeof(PJRT_Event_Await_Args), nullptr, handle_};
  return pjrt_error_to_status(GetPjrtApi()->pjrt_event_await(&args));
}

Client::Client(PJRT_Client *handle) noexcept : handle_(handle) {}

Client::~Client() noexcept {
  PJRT_Client_Destroy_Args args{sizeof(PJRT_Client_Destroy_Args), nullptr,
                                handle_};
  (void)GetPjrtApi()->pjrt_client_destroy(&args);
}

Client::Client(Client &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

Client &Client::operator=(Client &&other) noexcept {
  PJRT_Client *next = other.handle_;
  other.handle_ = nullptr;
  PJRT_Client_Destroy_Args args{sizeof(PJRT_Client_Destroy_Args), nullptr,
                                handle_};
  (void)GetPjrtApi()->pjrt_client_destroy(&args);
  handle_ = next;
  return *this;
}

Status Client::create(Client *client_out) noexcept {
  PJRT_Client_Create_Args args{sizeof(PJRT_Client_Create_Args),
                               nullptr,
                               nullptr,
                               0,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_create(&args);
  *client_out = Client(args.client);
  return pjrt_error_to_status(err);
}

int32_t Client::device_count() const noexcept {
  if (handle_ == nullptr) {
    return -1;
  }
  PJRT_Client_Devices_Args args{sizeof(PJRT_Client_Devices_Args), nullptr,
                                handle_, nullptr, 0};
  (void)GetPjrtApi()->pjrt_client_addressable_devices(&args);
  return static_cast<int32_t>(args.num_devices);
}

size_t Client::device_attribute_count(uint32_t device_id) const noexcept {
  PJRT_Client_LookupDevice_Args look_args{
      sizeof(PJRT_Client_LookupDevice_Args), nullptr, handle_,
      static_cast<int32_t>(device_id),       0,       nullptr};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_lookup_device(&look_args);
  if (err != nullptr) {
    (void)pjrt_error_to_status(err);
    return 0;
  }
  PJRT_Device_GetAttributes_Args attr_args{
      sizeof(PJRT_Device_GetAttributes_Args),
      nullptr,
      look_args.device,
      nullptr,
      0,
      0,
      0};
  (void)GetPjrtApi()->pjrt_device_get_attributes(&attr_args);
  return attr_args.num_attributes;
}

Status Client::buffer_from_host(uint32_t device_id, DataType dtype,
                                const int64_t *dims, size_t num_dims,
                                const uint8_t *host_data, size_t host_data_len,
                                Buffer *buffer_out) noexcept {
  (void)host_data_len;
  PJRT_Client_LookupDevice_Args look_args{
      sizeof(PJRT_Client_LookupDevice_Args), nullptr, handle_,
      static_cast<int32_t>(device_id),       0,       nullptr};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_lookup_device(&look_args);
  if (err == nullptr) {
    PJRT_Client_BufferFromHostBuffer_Args h2d_args{
        sizeof(PJRT_Client_BufferFromHostBuffer_Args),
        nullptr,
        handle_,
        host_data,
        static_cast<int32_t>(dtype),
        0,
        dims,
        num_dims,
        nullptr,
        0,
        0,
        0,
        look_args.device,
        nullptr,
        nullptr,
        nullptr,
        nullptr};
    (void)GetPjrtApi()->pjrt_client_buffer_from_host_buffer(&h2d_args);
    *buffer_out = Buffer(h2d_args.buffer);
  }
  return pjrt_error_to_status(err);
}

Status Client::compile(const char *format, const uint8_t *code, size_t code_len,
                       LoadedExecutable *exec_out) noexcept {
  size_t fmt_len = 0;
  while (format[fmt_len] != '\0') {
    ++fmt_len;
  }
  PJRT_Program prog{sizeof(PJRT_Program),
                    nullptr,
                    reinterpret_cast<const char *>(code),
                    code_len,
                    format,
                    fmt_len};
  PJRT_Client_Compile_Args comp_args{sizeof(PJRT_Client_Compile_Args),
                                     nullptr,
                                     handle_,
                                     &prog,
                                     nullptr,
                                     0,
                                     nullptr};
  PJRT_Error *err = GetPjrtApi()->pjrt_client_compile(&comp_args);
  if (err == nullptr) {
    *exec_out = LoadedExecutable(comp_args.executable);
  }
  return pjrt_error_to_status(err);
}

Status Client::deserialize_executable(const uint8_t *bytes, size_t bytes_len,
                                      LoadedExecutable *exec_out) noexcept {
  PJRT_Executable_DeserializeAndLoad_Args deser_args{
      sizeof(PJRT_Executable_DeserializeAndLoad_Args),
      nullptr,
      handle_,
      reinterpret_cast<const char *>(bytes),
      bytes_len,
      nullptr,
      nullptr,
      0};
  PJRT_Error *err =
      GetPjrtApi()->pjrt_executable_deserialize_and_load(&deser_args);
  if (err == nullptr) {
    *exec_out = LoadedExecutable(deser_args.loaded_executable);
  }
  return pjrt_error_to_status(err);
}

PJRT_Client *Client::raw() const noexcept { return handle_; }

Profiler::Profiler() noexcept : handle_(nullptr), status_(TpuStatus_New()) {
  TpuProfiler_Create(&handle_, status_);
}

Profiler::~Profiler() noexcept {
  TpuProfiler_Destroy(handle_);
  TpuStatus_Free(status_);
}

Profiler::Profiler(Profiler &&other) noexcept
    : handle_(other.handle_), status_(other.status_) {
  other.handle_ = nullptr;
  other.status_ = nullptr;
}

Profiler &Profiler::operator=(Profiler &&other) noexcept {
  TpuProfiler *next_handle = other.handle_;
  TpuStatus *next_status = other.status_;
  other.handle_ = nullptr;
  other.status_ = nullptr;
  TpuProfiler_Destroy(handle_);
  TpuStatus_Free(status_);
  handle_ = next_handle;
  status_ = next_status;
  return *this;
}

bool Profiler::start() noexcept {
  TpuProfiler_Start(handle_, status_);
  return TpuStatus_Ok(status_);
}

bool Profiler::stop() noexcept {
  TpuProfiler_Stop(handle_, status_);
  return TpuStatus_Ok(status_);
}

size_t Profiler::collect_size() noexcept {
  size_t sz = 0;
  TpuProfiler_CollectData(handle_, status_, nullptr, &sz);
  return sz;
}

Topology::Topology() noexcept : handle_(TpuUtil_GetTopologyPtr()) {}

int32_t Topology::logical_devices_per_host() const noexcept {
  return TpuTopology_LogicalDevicesPerHost(handle_, 0);
}

int32_t Topology::chip_bounds_x() const noexcept {
  return TpuTopology_ChipBounds_X(handle_);
}

int32_t Topology::chip_bounds_y() const noexcept {
  return TpuTopology_ChipBounds_Y(handle_);
}

int32_t Topology::chip_bounds_z() const noexcept {
  return TpuTopology_ChipBounds_Z(handle_);
}

int32_t Topology::version() const noexcept {
  return TpuTopology_Version(handle_);
}

SdkClient::SdkClient() noexcept : api_(GetLibtpuSdkApi()), handle_(nullptr) {
  LibtpuSdk_Client_Args args{nullptr};
  (void)api_->client_create(&args);
  handle_ = args.client;
}

SdkClient::~SdkClient() noexcept {
  LibtpuSdk_Client_Args args{handle_};
  (void)api_->client_destroy(&args);
}

SdkClient::SdkClient(SdkClient &&other) noexcept
    : api_(other.api_), handle_(other.handle_) {
  other.handle_ = nullptr;
}

SdkClient &SdkClient::operator=(SdkClient &&other) noexcept {
  LibtpuSdk_Client *next = other.handle_;
  other.handle_ = nullptr;
  LibtpuSdk_Client_Args args{handle_};
  (void)api_->client_destroy(&args);
  handle_ = next;
  return *this;
}

size_t SdkClient::core_state_summary_size() const noexcept {
  LibtpuSdk_GetRuntimeStatus_Args rstat_args{
      handle_, true, {0, 0, 0, 0, 0, 0, 0}, nullptr};
  (void)api_->get_runtime_status(&rstat_args);
  LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args sum_args{rstat_args.status,
                                                            nullptr, 0};
  (void)api_->runtime_status_get_core_state_summary(&sum_args);
  LibtpuSdk_RuntimeStatus_Destroy_Args del_args{rstat_args.status};
  (void)api_->runtime_status_destroy(&del_args);
  return sum_args.summary_size;
}

size_t SdkClient::metric_value_count(const char *metric_name) const noexcept {
  LibtpuSdk_GetMetric_Args m_args{handle_, metric_name, nullptr};
  (void)api_->get_metric(&m_args);
  LibtpuSdk_GetMetricValues_Args v_args{m_args.metric, nullptr, 0};
  (void)api_->get_metric_values(&v_args);
  return v_args.num_values;
}

size_t SdkClient::chip_count() const noexcept {
  LibtpuSdk_GetChipCoordinates_Args c_args{handle_, nullptr, 0};
  (void)api_->get_chip_coordinates(&c_args);
  return c_args.num_coords;
}

int64_t SdkClient::register_debug_callback(const void *callback,
                                           void *user_data) noexcept {
  LibtpuSdk_RegisterDebugCallback_Args args{handle_, callback, user_data, 0};
  (void)api_->register_debug_callback(&args);
  return args.callback_id;
}

bool SdkClient::unregister_debug_callback(int64_t callback_id) noexcept {
  LibtpuSdk_UnregisterDebugCallback_Args args{handle_, callback_id};
  LibtpuSdk_Error *err = api_->unregister_debug_callback(&args);
  if (err != nullptr) {
    struct DestroyArgs {
      LibtpuSdk_Error *error;
    } d_args{err};
    api_->error_destroy(&d_args);
    return false;
  }
  return true;
}

size_t SdkClient::trigger_debug_callbacks() const noexcept {
  return LibtpuSdk_TriggerDebugCallbacks(handle_);
}

PjrtApiView::PjrtApiView() noexcept : api_(GetPjrtApi()) {}

size_t PjrtApiView::struct_size() const noexcept { return api_->struct_size; }

int32_t PjrtApiView::minor_version() const noexcept {
  return api_->pjrt_api_version.minor_version;
}

size_t PjrtApiView::extension_count() const noexcept {
  size_t count = 0;
  const PJRT_Extension_Base *cur = api_->extension_start;
  while (cur != nullptr) {
    ++count;
    cur = cur->next;
  }
  return count;
}

const PJRT_Extension_Base *
PjrtApiView::find_extension(int32_t ext_type) const noexcept {
  const PJRT_Extension_Base *cur = api_->extension_start;
  while (cur != nullptr) {
    if (cur->ext_type == ext_type) {
      return cur;
    }
    cur = cur->next;
  }
  return nullptr;
}

size_t PjrtApiView::memory_description_count() const noexcept {
  PJRT_DeviceDescription_MemoryDescriptions_Args args{
      sizeof(PJRT_DeviceDescription_MemoryDescriptions_Args),
      nullptr,
      nullptr,
      nullptr,
      0,
      -1};
  (void)PJRT_MemoryDescriptions_Get(&args);
  return args.num_memory_descriptions;
}

Device::Device(PJRT_Device *handle) noexcept : handle_(handle) {}

PJRT_Device *Device::raw() const noexcept { return handle_; }

Memory::Memory(PJRT_Memory *handle) noexcept : handle_(handle) {}

PJRT_Memory *Memory::raw() const noexcept { return handle_; }

Executable::Executable(PJRT_Executable *handle) noexcept : handle_(handle) {}

PJRT_Executable *Executable::raw() const noexcept { return handle_; }

PhaseCompiler::PhaseCompiler() noexcept : handle_(0) {
  PJRT_PhaseCompile_GetCompiler_Args args{
      sizeof(PJRT_PhaseCompile_GetCompiler_Args), nullptr, nullptr, 0};
  (void)PJRT_PhaseCompile_GetCompiler(&args);
  handle_ = args.compiler_out;
}

PhaseCompiler::~PhaseCompiler() noexcept {
  PJRT_PhaseCompile_Destroy_Args args{sizeof(PJRT_PhaseCompile_Destroy_Args),
                                      nullptr,
                                      reinterpret_cast<void *>(handle_)};
  (void)PJRT_PhaseCompile_DestroyCompiler(&args);
}

size_t PhaseCompiler::phase_count() const noexcept {
  PJRT_PhaseCompile_GetPhaseNames_Args args{
      sizeof(PJRT_PhaseCompile_GetPhaseNames_Args),
      nullptr,
      reinterpret_cast<const void *>(handle_),
      nullptr,
      nullptr,
      0};
  (void)PJRT_PhaseCompile_GetPhaseNames(&args);
  return args.num_phase_names;
}

size_t PhaseCompiler::run_phase(const char *phase_name, const char *input_bytes,
                                size_t input_size) noexcept {
  PJRT_PhaseCompile_RunPhase_Args args{sizeof(PJRT_PhaseCompile_RunPhase_Args),
                                       nullptr,
                                       reinterpret_cast<void *>(handle_),
                                       input_bytes,
                                       input_size,
                                       phase_name,
                                       0};
  (void)PJRT_PhaseCompile_RunPhase(&args);
  return args.output_status;
}

MegascaleContext::MegascaleContext() noexcept : handle_(0) {
  PJRT_Megascale_Query_Args args{sizeof(PJRT_Megascale_Query_Args), nullptr, 0,
                                 0};
  (void)PJRT_Megascale_CreateClientContext(&args);
  handle_ = args.out_value;
}

MegascaleContext::~MegascaleContext() noexcept {
  PJRT_Megascale_Handle_Args args{sizeof(PJRT_Megascale_Handle_Args), nullptr,
                                  reinterpret_cast<void *>(handle_)};
  (void)PJRT_Megascale_DeleteClientContext(&args);
}

size_t MegascaleContext::barrier() noexcept {
  PJRT_Megascale_Query_Args args{sizeof(PJRT_Megascale_Query_Args), nullptr,
                                 handle_, 0};
  (void)PJRT_Megascale_Barrier(&args);
  return args.out_value;
}

size_t MegascaleContext::num_slices() const noexcept {
  PJRT_Megascale_Query_Args args{sizeof(PJRT_Megascale_Query_Args), nullptr,
                                 handle_, 0};
  (void)PJRT_Megascale_GetNumSlices(&args);
  return args.out_value;
}

EmbeddingEngine::EmbeddingEngine() noexcept
    : state_(TpuEmbeddingEngineState_Create()) {}

EmbeddingEngine::~EmbeddingEngine() noexcept {
  TpuEmbeddingEngineState_Free(state_);
}

const void *EmbeddingEngine::raw_state() const noexcept {
  return TpuEmbeddingEngineState_GetState(state_);
}

Communicator::Communicator() noexcept : handle_(nullptr) {
  PJRT_Collectives_CreateCommunicator_Args args{
      sizeof(PJRT_Collectives_CreateCommunicator_Args),
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr};
  (void)PJRT_Collectives_CreateCommunicator(&args);
  handle_ = args.communicator;
}

Communicator::~Communicator() noexcept {
  PJRT_Communicator_Destroy_Args args{sizeof(PJRT_Communicator_Destroy_Args),
                                      nullptr, handle_};
  (void)PJRT_Communicator_Destroy(&args);
}

Status Communicator::all_reduce(const Buffer &send_buf, Buffer *recv_buf,
                                size_t count) noexcept {
  PJRT_Communicator_AllReduce_Args args{
      sizeof(PJRT_Communicator_AllReduce_Args),
      nullptr,
      handle_,
      const_cast<PJRT_Buffer *>(send_buf.raw()),
      const_cast<PJRT_Buffer *>(recv_buf->raw()),
      0,
      0,
      count,
      nullptr};
  return pjrt_error_to_status(PJRT_Communicator_AllReduce(&args));
}

} // namespace libtpu
