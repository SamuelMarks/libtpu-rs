/**
 * @file test_libtpu.cpp
 * @brief Comprehensive C++20 unit test suite achieving 100% function, line, and
 * branch coverage of `libtpu_cpp`.
 */

#include "libtpu.hpp"
#include <cstdint>
#include <cstring>

/**
 * @brief C debug callback trampoline incrementing an `int32_t` counter at
 * `user_data`.
 * @param user_data Pointer to `int32_t` counter.
 */
extern "C" void test_debug_callback(void *user_data) {
  int32_t *counter = static_cast<int32_t *>(user_data);
  *counter += 1;
}

/**
 * @brief C PJRT event readiness callback incrementing an `int32_t` counter at
 * `user_arg`.
 * @param error Optional `PJRT_Error*` status (`nullptr` on success).
 * @param user_arg Pointer to `int32_t` counter.
 */
extern "C" void test_event_ready_callback(PJRT_Error *error, void *user_arg) {
  (void)error;
  int32_t *counter = static_cast<int32_t *>(user_arg);
  *counter += 1;
}

/**
 * @brief C OpenXLA Typed FFI v0.2 handler callback returning `nullptr` (`OK`).
 * @param frame Pointer to `XLA_FFI_CallFrame`.
 * @return `nullptr` on success.
 */
extern "C" PJRT_Error *test_ffi_handler(XLA_FFI_CallFrame *frame) {
  (void)frame;
  return nullptr;
}

/**
 * @brief Main test entrypoint exercising all C and C++20 RAII methods and error
 * branches.
 * @return `0` on success, non-zero if any assertion fails.
 */
int main() {
  // 1. Verify C ABI PJRT vtable header
  const PJRT_Api *api = GetPjrtApi();
  if (api == nullptr || api->struct_size != 1144 ||
      api->pjrt_api_version.minor_version != 116) {
    return 1;
  }
  TfTpu_Initialize(true, 0, nullptr);

  // 2. Verify Status move constructor, move assignment, and message branches
  LibTpuCStatus raw_ok{0, nullptr};
  libtpu::Status st1(raw_ok);
  if (!st1.is_ok() || st1.code() != 0 || std::strlen(st1.message()) != 0) {
    return 2;
  }
  libtpu::Status st2(static_cast<libtpu::Status &&>(st1));
  st1 = static_cast<libtpu::Status &&>(st2);
  if (!st1.is_ok()) {
    return 3;
  }

  // 3. Verify Client lifecycle, move semantics, and uninitialized error
  // branches
  libtpu::Client empty_client;
  if (empty_client.device_count() != -1) {
    return 4;
  }

  libtpu::Client client;
  libtpu::Status create_st = libtpu::Client::create(&client);
  if (!create_st.is_ok() || client.device_count() != 4) {
    return 5;
  }
  libtpu::Client moved_client(static_cast<libtpu::Client &&>(client));
  client = static_cast<libtpu::Client &&>(moved_client);
  if (client.device_count() != 4 || client.device_attribute_count(0) != 3 ||
      client.device_attribute_count(99) != 0) {
    return 6;
  }

  // 3b. Verify Event lifecycle, move semantics, deferred on_ready, set, and
  // await
  libtpu::Event ev;
  libtpu::Status ev_st = libtpu::Event::create(&ev);
  libtpu::Event ev_moved(static_cast<libtpu::Event &&>(ev));
  ev = static_cast<libtpu::Event &&>(ev_moved);
  int32_t ev_cb_hits = 0;
  if (!ev_st.is_ok() || ev.is_ready() ||
      !ev.on_ready(&test_event_ready_callback, &ev_cb_hits).is_ok() ||
      ev_cb_hits != 0) {
    return 60;
  }
  if (!ev.set(0, nullptr).is_ok() || !ev.is_ready() || ev_cb_hits != 1 ||
      !ev.await().is_ok()) {
    return 61;
  }

  // 4. Verify Buffer lifecycle, move semantics, and error branches
  const int64_t dims[1] = {4};
  const float host_f0[4] = {10.0F, 20.0F, 30.0F, 40.0F};
  const float host_f1[4] = {1.0F, 2.0F, 3.0F, 4.0F};

  libtpu::Buffer b0;
  libtpu::Buffer b1;
  libtpu::Status b0_st = client.buffer_from_host(
      0, libtpu::DataType::F32, dims, 1,
      reinterpret_cast<const uint8_t *>(host_f0), sizeof(host_f0), &b0);
  libtpu::Status b1_st = client.buffer_from_host(
      0, libtpu::DataType::F32, dims, 1,
      reinterpret_cast<const uint8_t *>(host_f1), sizeof(host_f1), &b1);
  if (!b0_st.is_ok() || !b1_st.is_ok() || b0.raw() == nullptr) {
    return 7;
  }

  // Error branch on buffer_from_host (invalid device 99)
  libtpu::Buffer b_err;
  libtpu::Status b_err_st = client.buffer_from_host(
      99, libtpu::DataType::F32, dims, 1,
      reinterpret_cast<const uint8_t *>(host_f0), sizeof(host_f0), &b_err);
  if (b_err_st.is_ok() || std::strlen(b_err_st.message()) == 0) {
    return 8;
  }

  libtpu::Buffer b0_moved(static_cast<libtpu::Buffer &&>(b0));
  b0 = static_cast<libtpu::Buffer &&>(b0_moved);
  if (b0.rank() != 1 || b0.dimension(0) != 4 || b0.dimension(99) != -1 ||
      b0.element_type() != libtpu::DataType::F32 || b0.tile_count() != 1) {
    return 80;
  }
  libtpu::Buffer b_copied;
  libtpu::Buffer b1_copied;
  libtpu::Status copy_ok = b0.copy_to_device(client.raw(), 1, &b_copied);
  libtpu::Status copy1_ok = b1.copy_to_device(client.raw(), 1, &b1_copied);
  libtpu::Status copy_err = b0.copy_to_device(client.raw(), 99, &b_copied);
  if (!copy_ok.is_ok() || !copy1_ok.is_ok() || copy_err.is_ok()) {
    return 81;
  }

  // 5. Verify LoadedExecutable compilation, move semantics, execution, and
  // error branches
  const char *mlir_code = "module @add_f32 { stablehlo.add }";
  libtpu::LoadedExecutable exec;
  libtpu::Status comp_st =
      client.compile("mlir", reinterpret_cast<const uint8_t *>(mlir_code),
                     std::strlen(mlir_code), &exec);
  if (!comp_st.is_ok() || exec.raw() == nullptr) {
    return 9;
  }

  // Error branch on compile (invalid format)
  libtpu::LoadedExecutable bad_exec;
  libtpu::Status bad_comp_st = client.compile(
      "invalid_format", reinterpret_cast<const uint8_t *>(mlir_code),
      std::strlen(mlir_code), &bad_exec);
  if (bad_comp_st.is_ok()) {
    return 10;
  }

  libtpu::LoadedExecutable exec_moved(
      static_cast<libtpu::LoadedExecutable &&>(exec));
  exec = static_cast<libtpu::LoadedExecutable &&>(exec_moved);
  char fp_buf[40] = {};
  char short_fp_buf[4] = {};
  if (exec.fingerprint(fp_buf, sizeof(fp_buf)) != 32 ||
      exec.fingerprint(short_fp_buf, sizeof(short_fp_buf)) != 32 ||
      exec.flops() < 4.0F || exec.num_outputs() != 1) {
    return 100;
  }
  size_t ser_len = 0;
  uint8_t ser_buf[4096] = {};
  uint8_t short_ser_buf[4] = {};
  if (!exec.serialize(ser_buf, sizeof(ser_buf), &ser_len).is_ok() ||
      ser_len == 0 ||
      exec.serialize(short_ser_buf, sizeof(short_ser_buf), &ser_len).is_ok()) {
    return 101;
  }
  size_t opt_len = 0;
  uint8_t opt_buf[256] = {};
  if (!exec.optimized_program(opt_buf, sizeof(opt_buf), &opt_len).is_ok() ||
      opt_len == 0 || exec.optimized_program(opt_buf, 1, &opt_len).is_ok()) {
    return 103;
  }
  libtpu::LoadedExecutable deser_exec;
  libtpu::Status deser_ok =
      client.deserialize_executable(ser_buf, ser_len, &deser_exec);
  libtpu::Status deser_err = client.deserialize_executable(
      reinterpret_cast<const uint8_t *>("bad"), 3, &deser_exec);
  if (!deser_ok.is_ok() || deser_err.is_ok()) {
    return 102;
  }

  // Execute success branch
  libtpu::Buffer out_buf;
  libtpu::Status exec_st = exec.execute_binary(0, b0, b1, &out_buf);
  if (!exec_st.is_ok()) {
    return 11;
  }

  const PJRT_Buffer *raw_inputs[2] = {b_copied.raw(), b1_copied.raw()};
  PJRT_Buffer *raw_outputs[1] = {nullptr};
  libtpu::Status multi_st = exec.execute(1, raw_inputs, 2, raw_outputs, 1);
  if (!multi_st.is_ok() || raw_outputs[0] == nullptr ||
      exec.execute(99, raw_inputs, 2, raw_outputs, 1).is_ok()) {
    return 110;
  }
  libtpu_rs_buffer_destroy(raw_outputs[0]);

  const PJRT_Buffer *dev0_inputs[2] = {b0.raw(), b1.raw()};
  const PJRT_Buffer *dev1_inputs[2] = {b_copied.raw(), b1_copied.raw()};
  const PJRT_Buffer *const *sharded_args[2] = {dev0_inputs, dev1_inputs};
  PJRT_Buffer *dev0_outs[1] = {nullptr};
  PJRT_Buffer *dev1_outs[1] = {nullptr};
  PJRT_Buffer *const sharded_outs[2] = {dev0_outs[0], dev1_outs[0]};
  PJRT_Buffer **sharded_out_lists[2] = {dev0_outs, dev1_outs};
  (void)sharded_outs;
  if (!exec.execute_sharded(sharded_args, 2, 2, sharded_out_lists).is_ok() ||
      dev0_outs[0] == nullptr || dev1_outs[0] == nullptr ||
      exec.cost_analysis_property_count() != 3) {
    return 111;
  }
  libtpu_rs_buffer_destroy(dev0_outs[0]);
  libtpu_rs_buffer_destroy(dev1_outs[0]);

  // Execute error branch (invalid device 99)
  libtpu::Buffer bad_out;
  libtpu::Status bad_exec_st = exec.execute_binary(99, b0, b1, &bad_out);
  if (bad_exec_st.is_ok()) {
    return 12;
  }

  float host_out[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  libtpu::Status read_st =
      out_buf.to_host(reinterpret_cast<uint8_t *>(host_out), sizeof(host_out));
  if (!read_st.is_ok() || host_out[0] != 11.0F || host_out[3] != 44.0F) {
    return 13;
  }

  // 6. Verify Profiler and Topology RAII wrappers
  static_assert(sizeof(PJRT_Api) == 1144, "PJRT_Api must be 1144 bytes");
  static_assert(sizeof(LibtpuSdk_Api) == 144,
                "LibtpuSdk_Api must be 144 bytes");
  static_assert(sizeof(TpuDriverFn) == 224, "TpuDriverFn must be 224 bytes");
  const LibtpuSdk_Api *sdk_api = GetLibtpuSdkApi();
  if (sdk_api == nullptr || sdk_api->version_header != 0x0000000100000000ULL) {
    return 14;
  }
  libtpu::SdkClient sdk_client;
  libtpu::SdkClient sdk_moved(static_cast<libtpu::SdkClient &&>(sdk_client));
  sdk_client = static_cast<libtpu::SdkClient &&>(sdk_moved);
  if (sdk_client.core_state_summary_size() == 0 ||
      sdk_client.metric_value_count("tpu.runtime.hbm.usage_bytes") != 4 ||
      sdk_client.chip_count() != 4) {
    return 140;
  }
  int32_t cb_hits = 0;
  int64_t cb_id = sdk_client.register_debug_callback(
      reinterpret_cast<const void *>(&test_debug_callback), &cb_hits);
  if (cb_id <= 0 || sdk_client.trigger_debug_callbacks() != 1 || cb_hits != 1) {
    return 141;
  }
  if (!sdk_client.unregister_debug_callback(cb_id) ||
      sdk_client.unregister_debug_callback(cb_id)) {
    return 142;
  }

  libtpu::Topology topo;
  if (topo.logical_devices_per_host() != 4 || topo.chip_bounds_x() != 2 ||
      topo.chip_bounds_y() != 2 || topo.chip_bounds_z() != 1 ||
      topo.version() != 5) {
    return 15;
  }

  libtpu::Profiler prof;
  libtpu::Profiler prof_moved(static_cast<libtpu::Profiler &&>(prof));
  prof = static_cast<libtpu::Profiler &&>(prof_moved);
  if (!prof.start()) {
    return 16;
  }
  if (!prof.stop()) {
    return 17;
  }
  if (prof.collect_size() == 0) {
    return 18;
  }

  // 7. Verify legacy TensorFlow, SparseCore, and Embedding C ABI structures
  static_assert(sizeof(TFNPD_PluginParams) == 40,
                "TFNPD_PluginParams must be 40 bytes");
  static_assert(sizeof(TpuPartitionedCall_Params) == 24,
                "TpuPartitionedCall_Params must be 24 bytes");
  static_assert(sizeof(XLA_Shape) == 96, "XLA_Shape must be 96 bytes");
  static_assert(sizeof(TpuEmbeddingEngine_Params) == 56,
                "TpuEmbeddingEngine_Params must be 56 bytes");

  TFNPD_PluginParams tfnpd{sizeof(TFNPD_PluginParams),
                           nullptr,
                           nullptr,
                           nullptr,
                           0,
                           false,
                           false,
                           0};
  TFNPD_InitPlugin(&tfnpd, nullptr);
  if (tfnpd.priority != 55 || !tfnpd.is_pluggable_device) {
    return 19;
  }

  TpuPartitionedCall_Params pcall{};
  if (TfTpu_GetTpuPartitionedCallParams(&pcall) != 0 ||
      !pcall.input_shape_opt) {
    return 20;
  }

  const int64_t shape_dims[2] = {8, 128};
  XLA_Shape host_shape{11, 2, shape_dims, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0}};
  XLA_Shape infeed_shape{};
  if (TpuTransferManager_GetInfeedLayout(&host_shape, &infeed_shape) != 0 ||
      infeed_shape.tile_dims[1] != 128) {
    return 21;
  }

  int64_t max_ids = 0;
  int64_t max_uniques = 0;
  if (SparseCore_GetMaxIdsAndUniques("vocab", 32, 16, &max_ids, &max_uniques,
                                     nullptr) != 128 ||
      max_ids < 128) {
    return 22;
  }

  size_t emb_sz = 0;
  char *emb_out = nullptr;
  TpuEmbeddingEngine_Params emb_params{sizeof(TpuEmbeddingEngine_Params),
                                       nullptr,
                                       nullptr,
                                       0,
                                       &emb_sz,
                                       &emb_out,
                                       nullptr};
  if (TpuEmbeddingEngine_ExecutePartitioner(&emb_params) != 0 || emb_sz == 0 ||
      emb_out == nullptr) {
    return 23;
  }
  TpuConfigurationApi_FreeCharArray(emb_out);

  TpuDriverFn drv_fn{};
  TpuDriver_Initialize(&drv_fn, true);
  if (drv_fn.open == nullptr || drv_fn.version == nullptr) {
    return 24;
  }

  libtpu::PjrtApiView api_view;
  PJRT_DeviceDescription_MemoryDescriptions_Args mdesc_args{
      sizeof(PJRT_DeviceDescription_MemoryDescriptions_Args),
      nullptr,
      nullptr,
      nullptr,
      0,
      -1};
  (void)PJRT_MemoryDescriptions_Get(&mdesc_args);
  PJRT_MemoryDescription_Kind_Args kind_args{
      sizeof(PJRT_MemoryDescription_Kind_Args),
      nullptr,
      mdesc_args.memory_descriptions[1],
      nullptr,
      0,
      -1,
      0};
  (void)PJRT_MemoryDescription_Kind(&kind_args);
  if (api_view.struct_size() != 1144 || api_view.minor_version() != 116 ||
      api_view.extension_count() != 17 ||
      api_view.memory_description_count() != 4 || kind_args.kind_id != 1 ||
      api_view.find_extension(16) == nullptr ||
      api_view.find_extension(999) != nullptr) {
    return 25;
  }

  libtpu::Device dev_view(nullptr);
  libtpu::Memory mem_view(nullptr);
  libtpu::Executable exec_view(nullptr);
  libtpu::PhaseCompiler phase_compiler;
  libtpu::PhaseCompiler phase_moved(
      static_cast<libtpu::PhaseCompiler &&>(phase_compiler));
  phase_compiler = static_cast<libtpu::PhaseCompiler &&>(phase_moved);
  libtpu::MegaScaleContext ms_ctx;
  libtpu::MegaScaleContext ms_moved(
      static_cast<libtpu::MegaScaleContext &&>(ms_ctx));
  ms_ctx = static_cast<libtpu::MegaScaleContext &&>(ms_moved);
  libtpu::EmbeddingEngine emb_engine;
  libtpu::CollectivesCommunicator comm;
  libtpu::CollectivesCommunicator comm_moved(
      static_cast<libtpu::CollectivesCommunicator &&>(comm));
  comm = static_cast<libtpu::CollectivesCommunicator &&>(comm_moved);
  float ar_out[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  if (dev_view.raw() != nullptr || mem_view.raw() != nullptr ||
      exec_view.raw() != nullptr || !comm.all_reduce(b0, &b1, 4).is_ok() ||
      !b1.to_host(reinterpret_cast<uint8_t *>(ar_out), sizeof(ar_out))
           .is_ok() ||
      ar_out[0] != 11.0F || ar_out[3] != 44.0F) {
    return 26;
  }
  if (phase_compiler.phase_count() != 5 ||
      phase_compiler.run_phase("hlo_verification", mlir_code,
                               std::strlen(mlir_code)) == 0) {
    return 27;
  }
  if (ms_ctx.barrier() == 0 || ms_ctx.num_slices() == 0 ||
      emb_engine.raw_state() == nullptr ||
      emb_engine.state_handle() == nullptr ||
      static_cast<int32_t>(libtpu::TpuVersion::V5e) != topo.version()) {
    return 28;
  }

  libtpu::Result<int32_t> ok_res(42);
  libtpu::Result<int32_t> err_res(libtpu::Status(LibTpuCStatus{
      static_cast<int32_t>(libtpu::ErrorCode::Internal), nullptr}));
  libtpu::CollectiveCommunicator coll_comm;
  libtpu::MultiSliceConfig ms_cfg;
  libtpu::SparseCoreEngine sc_eng;
  if (sc_eng.raw_state() == nullptr) {
    return 280;
  }
  const float emb_init[4] = {1.5F, 2.5F, 3.5F, 4.5F};
  float emb_read[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  const std::string part_cfg = emb_engine.partition_parameters("table_0", 7);
  const size_t emb_written =
      emb_engine.write_parameters("table_0", 7, emb_init, 4);
  const size_t emb_read_cnt =
      emb_engine.read_parameters("table_0", 7, emb_read, 4);
  if (!ok_res.is_ok() || ok_res.value() != 42 || !ok_res.status().is_ok() ||
      err_res.is_ok() || ms_cfg.num_slices() == 0 ||
      !coll_comm.all_reduce(b0, &b1, 4).is_ok() ||
      !coll_comm.reduce_scatter(b0, &b1, 4).is_ok() ||
      !coll_comm.all_gather(b0, &b1, 4).is_ok() ||
      !coll_comm.all_to_all(b0, &b1, 4).is_ok() ||
      !coll_comm.collective_permute(b0, &b1, 4).is_ok() ||
      coll_comm.to_string().empty() || !api_view.default_layout_allocated() ||
      api_view.create_raw_buffer_device_ptr(b0) == 0 ||
      api_view.cross_host_send(9) == 0 || part_cfg.empty() ||
      emb_written != 4 || emb_read_cnt != 4 || emb_read[0] != 1.5F ||
      emb_engine.build_load_optimizer_hlo() == 0) {
    return 29;
  }

  libtpu::AsyncTransferManager atm(client);
  libtpu::AsyncTransferManager atm_moved(
      static_cast<libtpu::AsyncTransferManager &&>(atm));
  atm = static_cast<libtpu::AsyncTransferManager &&>(atm_moved);
  libtpu::Buffer atm_buf;
  const int64_t nd_dims[2] = {2, 2};
  float nd_out[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  const int64_t s64_in0[2] = {0x7FFFFFFFFFFFFFFELL, 100LL};
  const int64_t s64_in1[2] = {1LL, 25LL};
  int64_t s64_out[2] = {0LL, 0LL};
  const int64_t s64_dims[1] = {2};
  const char *s64_mlir = "module @add_s64 {\n"
                         "  func.func @main(%a: tensor<2xi64>, %b: "
                         "tensor<2xi64>) -> tensor<2xi64> {\n"
                         "    %0 = stablehlo.add %a, %b : tensor<2xi64>\n"
                         "    func.return %0 : tensor<2xi64>\n"
                         "  }\n"
                         "}";
  if (atm.buffer_size(0) == 0 || atm.device() == nullptr ||
      !atm.transfer_data(0, host_f0, 0, sizeof(host_f0), true).is_ok() ||
      !atm.retrieve_buffer(0, &atm_buf).is_ok() || atm_buf.raw() == nullptr ||
      !client
           .compile_and_execute_nd(
               0, "mlir", reinterpret_cast<const uint8_t *>(mlir_code),
               std::strlen(mlir_code), libtpu::DataType::F32, nd_dims, 2,
               reinterpret_cast<const uint8_t *>(host_f0),
               reinterpret_cast<const uint8_t *>(host_f1), sizeof(host_f0),
               reinterpret_cast<uint8_t *>(nd_out))
           .is_ok() ||
      nd_out[0] != 11.0F || nd_out[3] != 44.0F ||
      !client
           .compile_and_execute_nd(
               0, "mlir", reinterpret_cast<const uint8_t *>(s64_mlir),
               std::strlen(s64_mlir), libtpu::DataType::S64, s64_dims, 1,
               reinterpret_cast<const uint8_t *>(s64_in0),
               reinterpret_cast<const uint8_t *>(s64_in1), sizeof(s64_in0),
               reinterpret_cast<uint8_t *>(s64_out))
           .is_ok() ||
      s64_out[0] != 0x7FFFFFFFFFFFFFFFLL || s64_out[1] != 125LL) {
    return 30;
  }

  libtpu::TpuExecutableInspector exec_insp(nullptr);
  libtpu::ShardingsInspector shard_insp(nullptr);
  const std::string tile_proto_schema =
      libtpu::emit_proto_schema("xla.TileProto");
  const std::string bad_proto_schema =
      libtpu::emit_proto_schema("nonexistent.Msg");
  if (exec_insp.target_arguments_size() == 0 ||
      exec_insp.hlo_module_with_config_size() == 0 ||
      exec_insp.generated_code_size_bytes() <= 0 ||
      exec_insp.set_and_query_compilation_env(1) == 0 ||
      shard_insp.parameter_sharding_count() == 0 ||
      shard_insp.output_sharding_count() == 0 || tile_proto_schema.empty() ||
      !bad_proto_schema.empty()) {
    return 31;
  }

  libtpu::Coordinator coord;
  libtpu::Coordinator coord_moved(static_cast<libtpu::Coordinator &&>(coord));
  coord = static_cast<libtpu::Coordinator &&>(coord_moved);
  if (!coord.put_key_value("topo/host0", "ready").is_ok() ||
      coord.get_key_value("topo/host0") != "ready" ||
      !coord.get_key_value("missing_key").empty()) {
    return 32;
  }

  libtpu::LoadedExecutable null_exec;
  const uint8_t empty_pb[1] = {0};
  const std::string llo_text = libtpu::disassemble_executable_llo(exec);
  const std::string bad_llo = libtpu::disassemble_executable_llo(null_exec);
  const std::string cd_json = libtpu::read_core_dump_json(empty_pb, 0);
  const std::string bad_cd_json = libtpu::read_core_dump_json(nullptr, 0);
  const std::string ss_json = libtpu::read_stall_summaries_json(empty_pb, 0);
  const std::string bad_ss_json = libtpu::read_stall_summaries_json(nullptr, 0);
  const std::string cm_json = libtpu::read_compiler_metadata_json(empty_pb, 0);
  const std::string bad_cm_json =
      libtpu::read_compiler_metadata_json(nullptr, 0);
  const std::string reg_summary = libtpu::inspect_register_allocation(exec);
  const std::string bad_reg_summary =
      libtpu::inspect_register_allocation(null_exec);
  const libtpu::Status ffi_ok =
      libtpu::register_ffi_handler("cpp_ffi_custom_op", &test_ffi_handler);
  const libtpu::Status ffi_bad =
      libtpu::register_ffi_handler(nullptr, &test_ffi_handler);
  const std::string chip_cfg_summary = libtpu::query_chip_config_summary();
  const uint32_t wrap_hops = libtpu::compute_ici_route(0, 0, 0, 3, 0, 0, true);
  const uint32_t mesh_hops = libtpu::compute_ici_route(0, 0, 0, 3, 0, 0, false);
  if (llo_text.empty() || !bad_llo.empty() || cd_json.empty() ||
      !bad_cd_json.empty() || ss_json.empty() || !bad_ss_json.empty() ||
      cm_json.empty() || !bad_cm_json.empty() || reg_summary.empty() ||
      !bad_reg_summary.empty() || !ffi_ok.is_ok() || ffi_bad.is_ok() ||
      chip_cfg_summary.empty() || wrap_hops != 1 || mesh_hops != 3) {
    return 33;
  }

  return 0;
}
