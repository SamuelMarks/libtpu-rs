/**
 * @file test_libtpu_c11.c
 * @brief Strict C11 (`-std=c11 -Wall -Wextra -Wpedantic -Wconversion -Werror`)
 * ABI offset/size verification and runtime test suite for `libtpu.h`.
 */

#include "libtpu.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/** @cond INTERNAL */
/* Bit-exact C11 offset verification across all 138 PJRT_Api function slots. */
_Static_assert(offsetof(PJRT_Api, struct_size) == 0x000, "struct_size");
_Static_assert(offsetof(PJRT_Api, extension_start) == 0x008, "extension_start");
_Static_assert(offsetof(PJRT_Api, pjrt_api_version) == 0x010, "api_version");
_Static_assert(offsetof(PJRT_Api, pjrt_error_destroy) == 0x028, "slot 1");
_Static_assert(offsetof(PJRT_Api, pjrt_error_message) == 0x030, "slot 2");
_Static_assert(offsetof(PJRT_Api, pjrt_error_get_code) == 0x038, "slot 3");
_Static_assert(offsetof(PJRT_Api, pjrt_plugin_initialize) == 0x040, "slot 4");
_Static_assert(offsetof(PJRT_Api, pjrt_plugin_attributes) == 0x048, "slot 5");
_Static_assert(offsetof(PJRT_Api, pjrt_event_destroy) == 0x050, "slot 6");
_Static_assert(offsetof(PJRT_Api, pjrt_event_is_ready) == 0x058, "slot 7");
_Static_assert(offsetof(PJRT_Api, pjrt_event_error) == 0x060, "slot 8");
_Static_assert(offsetof(PJRT_Api, pjrt_event_await) == 0x068, "slot 9");
_Static_assert(offsetof(PJRT_Api, pjrt_event_on_ready) == 0x070, "slot 10");
_Static_assert(offsetof(PJRT_Api, pjrt_client_create) == 0x078, "slot 11");
_Static_assert(offsetof(PJRT_Api, pjrt_client_destroy) == 0x080, "slot 12");
_Static_assert(offsetof(PJRT_Api, pjrt_client_platform_name) == 0x088,
               "slot 13");
_Static_assert(offsetof(PJRT_Api, pjrt_client_process_index) == 0x090,
               "slot 14");
_Static_assert(offsetof(PJRT_Api, pjrt_client_platform_version) == 0x098,
               "slot 15");
_Static_assert(offsetof(PJRT_Api, pjrt_client_devices) == 0x0a0, "slot 16");
_Static_assert(offsetof(PJRT_Api, pjrt_client_addressable_devices) == 0x0a8,
               "slot 17");
_Static_assert(offsetof(PJRT_Api, pjrt_client_lookup_device) == 0x0b0,
               "slot 18");
_Static_assert(offsetof(PJRT_Api, pjrt_client_lookup_addressable_device) ==
                   0x0b8,
               "slot 19");
_Static_assert(offsetof(PJRT_Api, pjrt_client_addressable_memories) == 0x0c0,
               "slot 20");
_Static_assert(offsetof(PJRT_Api, pjrt_client_compile) == 0x0c8, "slot 21");
_Static_assert(offsetof(PJRT_Api, pjrt_client_default_device_assignment) ==
                   0x0d0,
               "slot 22");
_Static_assert(offsetof(PJRT_Api, pjrt_client_buffer_from_host_buffer) == 0x0d8,
               "slot 23");
_Static_assert(offsetof(PJRT_Api, pjrt_device_description_id) == 0x0e0,
               "slot 24");
_Static_assert(offsetof(PJRT_Api, pjrt_device_description_process_index) ==
                   0x0e8,
               "slot 25");
_Static_assert(offsetof(PJRT_Api, pjrt_device_description_attributes) == 0x0f0,
               "slot 26");
_Static_assert(offsetof(PJRT_Api, pjrt_device_description_kind) == 0x0f8,
               "slot 27");
_Static_assert(offsetof(PJRT_Api, pjrt_device_description_debug_string) ==
                   0x100,
               "slot 28");
_Static_assert(offsetof(PJRT_Api, pjrt_device_description_to_string) == 0x108,
               "slot 29");
_Static_assert(offsetof(PJRT_Api, pjrt_device_get_description) == 0x110,
               "slot 30");
_Static_assert(offsetof(PJRT_Api, pjrt_device_is_addressable) == 0x118,
               "slot 31");
_Static_assert(offsetof(PJRT_Api, pjrt_device_local_hardware_id) == 0x120,
               "slot 32");
_Static_assert(offsetof(PJRT_Api, pjrt_device_addressable_memories) == 0x128,
               "slot 33");
_Static_assert(offsetof(PJRT_Api, pjrt_device_default_memory) == 0x130,
               "slot 34");
_Static_assert(offsetof(PJRT_Api, pjrt_device_memory_stats) == 0x138,
               "slot 35");
_Static_assert(offsetof(PJRT_Api, pjrt_memory_id) == 0x140, "slot 36");
_Static_assert(offsetof(PJRT_Api, pjrt_memory_kind) == 0x148, "slot 37");
_Static_assert(offsetof(PJRT_Api, pjrt_memory_debug_string) == 0x150,
               "slot 38");
_Static_assert(offsetof(PJRT_Api, pjrt_memory_to_string) == 0x158, "slot 39");
_Static_assert(offsetof(PJRT_Api, pjrt_memory_addressable_by_devices) == 0x160,
               "slot 40");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_destroy) == 0x168, "slot 41");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_name) == 0x170, "slot 42");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_num_replicas) == 0x178,
               "slot 43");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_num_partitions) == 0x180,
               "slot 44");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_num_outputs) == 0x188,
               "slot 45");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_executable_size_of_generated_code_in_bytes) ==
                   0x190,
               "slot 46");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_get_cost_analysis) == 0x198,
               "slot 47");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_output_memory_kinds) == 0x1a0,
               "slot 48");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_optimized_program) == 0x1a8,
               "slot 49");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_serialize) == 0x1b0,
               "slot 50");
_Static_assert(offsetof(PJRT_Api, pjrt_loaded_executable_destroy) == 0x1b8,
               "slot 51");
_Static_assert(offsetof(PJRT_Api, pjrt_loaded_executable_get_executable) ==
                   0x1c0,
               "slot 52");
_Static_assert(offsetof(PJRT_Api, pjrt_loaded_executable_addressable_devices) ==
                   0x1c8,
               "slot 53");
_Static_assert(offsetof(PJRT_Api, pjrt_loaded_executable_delete) == 0x1d0,
               "slot 54");
_Static_assert(offsetof(PJRT_Api, pjrt_loaded_executable_is_deleted) == 0x1d8,
               "slot 55");
_Static_assert(offsetof(PJRT_Api, pjrt_loaded_executable_execute) == 0x1e0,
               "slot 56");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_deserialize_and_load) ==
                   0x1e8,
               "slot 57");
_Static_assert(offsetof(PJRT_Api, pjrt_loaded_executable_fingerprint) == 0x1f0,
               "slot 58");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_destroy) == 0x1f8, "slot 59");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_element_type) == 0x200,
               "slot 60");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_dimensions) == 0x208, "slot 61");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_unpadded_dimensions) == 0x210,
               "slot 62");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_dynamic_dimension_indices) ==
                   0x218,
               "slot 63");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_get_memory_layout) == 0x220,
               "slot 64");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_on_device_size_in_bytes) == 0x228,
               "slot 65");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_device) == 0x230, "slot 66");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_memory) == 0x238, "slot 67");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_delete) == 0x240, "slot 68");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_is_deleted) == 0x248, "slot 69");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_copy_to_device) == 0x250,
               "slot 70");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_to_host_buffer) == 0x258,
               "slot 71");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_is_on_cpu) == 0x260, "slot 72");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_ready_event) == 0x268, "slot 73");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_unsafe_pointer) == 0x270,
               "slot 74");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_buffer_increase_external_reference_count) == 0x278,
               "slot 75");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_buffer_decrease_external_reference_count) == 0x280,
               "slot 76");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_buffer_opaque_device_memory_data_pointer) == 0x288,
               "slot 77");
_Static_assert(offsetof(PJRT_Api, pjrt_copy_to_device_stream_destroy) == 0x290,
               "slot 78");
_Static_assert(offsetof(PJRT_Api, pjrt_copy_to_device_stream_add_chunk) ==
                   0x298,
               "slot 79");
_Static_assert(offsetof(PJRT_Api, pjrt_copy_to_device_stream_total_bytes) ==
                   0x2a0,
               "slot 80");
_Static_assert(offsetof(PJRT_Api, pjrt_copy_to_device_stream_granule_size) ==
                   0x2a8,
               "slot 81");
_Static_assert(offsetof(PJRT_Api, pjrt_copy_to_device_stream_current_bytes) ==
                   0x2b0,
               "slot 82");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_create) == 0x2b8,
               "slot 83");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_destroy) == 0x2c0,
               "slot 84");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_platform_name) ==
                   0x2c8,
               "slot 85");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_platform_version) ==
                   0x2d0,
               "slot 86");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_topology_description_get_device_descriptions) ==
                   0x2d8,
               "slot 87");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_serialize) == 0x2e0,
               "slot 88");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_attributes) ==
                   0x2e8,
               "slot 89");
_Static_assert(offsetof(PJRT_Api, pjrt_compile) == 0x2f0, "slot 90");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_output_element_types) ==
                   0x2f8,
               "slot 91");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_output_dimensions) == 0x300,
               "slot 92");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_copy_to_memory) == 0x308,
               "slot 93");
_Static_assert(offsetof(PJRT_Api, pjrt_client_create_view_of_device_buffer) ==
                   0x310,
               "slot 94");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_fingerprint) == 0x318,
               "slot 95");
_Static_assert(offsetof(PJRT_Api, pjrt_client_topology_description) == 0x320,
               "slot 96");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_get_compiled_memory_stats) ==
                   0x328,
               "slot 97");
_Static_assert(offsetof(PJRT_Api, pjrt_memory_kind_id) == 0x330, "slot 98");
_Static_assert(offsetof(PJRT_Api, pjrt_execute_context_create) == 0x338,
               "slot 99");
_Static_assert(offsetof(PJRT_Api, pjrt_execute_context_destroy) == 0x340,
               "slot 100");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_copy_raw_to_host) == 0x348,
               "slot 101");
_Static_assert(offsetof(PJRT_Api, pjrt_async_transfer_manager_destroy) == 0x350,
               "slot 102");
_Static_assert(offsetof(PJRT_Api, pjrt_async_transfer_manager_transfer_data) ==
                   0x358,
               "slot 103");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_client_create_buffers_for_async_host_to_device) ==
                   0x360,
               "slot 104");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_async_transfer_manager_retrieve_buffer) == 0x368,
               "slot 105");
_Static_assert(offsetof(PJRT_Api, pjrt_async_transfer_manager_device) == 0x370,
               "slot 106");
_Static_assert(offsetof(PJRT_Api, pjrt_async_transfer_manager_buffer_count) ==
                   0x378,
               "slot 107");
_Static_assert(offsetof(PJRT_Api, pjrt_async_transfer_manager_buffer_size) ==
                   0x380,
               "slot 108");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_async_transfer_manager_set_buffer_error) == 0x388,
               "slot 109");
_Static_assert(offsetof(PJRT_Api, pjrt_async_transfer_manager_add_metadata) ==
                   0x390,
               "slot 110");
_Static_assert(offsetof(PJRT_Api, pjrt_client_dma_map) == 0x398, "slot 111");
_Static_assert(offsetof(PJRT_Api, pjrt_client_dma_unmap) == 0x3a0, "slot 112");
_Static_assert(offsetof(PJRT_Api, pjrt_client_create_uninitialized_buffer) ==
                   0x3a8,
               "slot 113");
_Static_assert(offsetof(PJRT_Api, pjrt_client_update_global_process_info) ==
                   0x3b0,
               "slot 114");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_deserialize) ==
                   0x3b8,
               "slot 115");
_Static_assert(offsetof(PJRT_Api, pjrt_client_create_alias_buffer) == 0x3c0,
               "slot 116");
_Static_assert(offsetof(PJRT_Api, pjrt_client_fulfill_alias_buffer) == 0x3c8,
               "slot 117");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_loaded_executable_get_device_assignment) == 0x3d0,
               "slot 118");
_Static_assert(offsetof(PJRT_Api, pjrt_client_create_error_buffer) == 0x3d8,
               "slot 119");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_async_transfer_manager_transfer_literal) == 0x3e0,
               "slot 120");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_copy_raw_to_host_future) == 0x3e8,
               "slot 121");
_Static_assert(offsetof(PJRT_Api, pjrt_device_poison_execution) == 0x3f0,
               "slot 122");
_Static_assert(offsetof(PJRT_Api, pjrt_device_create_async_tracking_event) ==
                   0x3f8,
               "slot 123");
_Static_assert(offsetof(PJRT_Api, pjrt_async_tracking_event_destroy) == 0x400,
               "slot 124");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_get_compile_options) == 0x408,
               "slot 125");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_donate_with_control_dependency) ==
                   0x410,
               "slot 126");
_Static_assert(offsetof(PJRT_Api, pjrt_event_create) == 0x418, "slot 127");
_Static_assert(offsetof(PJRT_Api, pjrt_event_set) == 0x420, "slot 128");
_Static_assert(offsetof(PJRT_Api, pjrt_device_get_attributes) == 0x428,
               "slot 129");
_Static_assert(offsetof(PJRT_Api, pjrt_client_load) == 0x430, "slot 130");
_Static_assert(
    offsetof(PJRT_Api, pjrt_loaded_executable_addressable_device_logical_ids) ==
        0x438,
    "slot 131");
_Static_assert(offsetof(PJRT_Api, pjrt_buffer_bitcast) == 0x440, "slot 132");
_Static_assert(offsetof(PJRT_Api, pjrt_error_for_each_payload) == 0x448,
               "slot 133");
_Static_assert(offsetof(PJRT_Api, pjrt_topology_description_fingerprint) ==
                   0x450,
               "slot 134");
_Static_assert(offsetof(PJRT_Api, pjrt_executable_parameter_memory_kinds) ==
                   0x458,
               "slot 135");
_Static_assert(offsetof(PJRT_Api, pjrt_device_clear_memory_stats) == 0x460,
               "slot 136");
_Static_assert(
    offsetof(PJRT_Api,
             pjrt_topology_description_make_canonical_shape_for_memory_space) ==
        0x468,
    "slot 137");
_Static_assert(offsetof(PJRT_Api,
                        pjrt_topology_description_get_memory_space_kind_ids) ==
                   0x470,
               "slot 138");
_Static_assert(offsetof(PJRT_Client_BufferFromHostBuffer_Args, device) == 0x50,
               "H2D.device == 0x50");
_Static_assert(offsetof(PJRT_Client_BufferFromHostBuffer_Args, memory) == 0x58,
               "H2D.memory == 0x58");
_Static_assert(offsetof(PJRT_Client_BufferFromHostBuffer_Args, device_layout) ==
                   0x60,
               "H2D.device_layout == 0x60");
_Static_assert(offsetof(PJRT_Client_BufferFromHostBuffer_Args,
                        done_with_host_buffer) == 0x68,
               "H2D.done_with_host_buffer == 0x68");
_Static_assert(offsetof(PJRT_Client_BufferFromHostBuffer_Args, buffer) == 0x70,
               "H2D.buffer == 0x70");
/** @endcond */

/**
 * @brief Entry point executing C11 vtable, extension chain, and end-to-end
 * compile/execute verification against `libtpu.h`.
 * @return `0` on full ABI and behavioral parity, non-zero on failure.
 */
int main(void) {
  const PJRT_Api *api = GetPjrtApi();
  if (api == NULL || api->struct_size != 1144U ||
      api->pjrt_api_version.minor_version != 117) {
    return 1;
  }

  /* Verify exact 18-node 0.0.53.1 TPU PJRT extension chain order, IDs, & sizes.
   */
  const int32_t expected_types[18] = {5,  24, 23, 22, 21, 20, 19, 18, 17,
                                      16, 14, 9,  12, 13, 6,  4,  8,  1};
  const size_t expected_sizes[18] = {48U, 56U,  32U, 64U,  96U, 120U,
                                     40U, 288U, 96U, 280U, 40U, 192U,
                                     56U, 40U,  40U, 80U,  80U, 40U};
  const PJRT_Extension_Base *cur = api->extension_start;
  for (size_t i = 0U; i < 18U; ++i) {
    if (cur == NULL || cur->ext_type != expected_types[i] ||
        cur->struct_size != expected_sizes[i]) {
      return 2;
    }
    cur = cur->next;
  }
  if (cur != NULL) {
    return 3;
  }

  const PJRT_FFI_Extension *ffi_ext = PJRT_FFI_GetExtension();
  const LibtpuSdk_Api *sdk = GetLibtpuSdkApi();
  if (ffi_ext == NULL || ffi_ext->base.ext_type != 5 ||
      ffi_ext->base.struct_size != 48U || sdk == NULL ||
      sdk->version_header != 0x0000000100000000ULL) {
    return 4;
  }

  /* Create PJRT_Client, exercise PJRT_XlaTransform_*, compile, and test ABI
   * compatibility. */
  PJRT_Client_Create_Args create_args;
  memset(&create_args, 0, sizeof(create_args));
  create_args.struct_size = sizeof(create_args);
  if (api->pjrt_client_create(&create_args) != NULL ||
      create_args.client == NULL) {
    return 5;
  }

  PJRT_XlaTransform_Register_Args xt_reg = {
      sizeof(PJRT_XlaTransform_Register_Args), NULL, "c11_test_pass", 13U, 77U};
  if (PJRT_XlaTransform_Register(&xt_reg) != NULL) {
    return 51;
  }
  PJRT_XlaTransform_List_Args xt_list = {sizeof(PJRT_XlaTransform_List_Args),
                                         NULL, NULL, 0U, NULL};
  if (PJRT_XlaTransform_List(&xt_list) != NULL || xt_list.num_names_out < 1U ||
      xt_list.names_out == NULL || xt_list.name_sizes_out == NULL) {
    return 52;
  }

  const char mlir_src[] =
      "module @c11 { func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> "
      "tensor<4xf32> { %0 = stablehlo.add %a, %b : tensor<4xf32> return %0 : "
      "tensor<4xf32> } }";
  PJRT_Program prog;
  memset(&prog, 0, sizeof(prog));
  prog.struct_size = sizeof(prog);
  prog.code = (const char *)(uintptr_t)mlir_src;
  prog.code_size = sizeof(mlir_src) - 1U;
  prog.format = "mlir";
  prog.format_size = 4U;

  PJRT_Client_Compile_Args comp_args;
  memset(&comp_args, 0, sizeof(comp_args));
  comp_args.struct_size = sizeof(comp_args);
  comp_args.client = create_args.client;
  comp_args.program = &prog;
  if (api->pjrt_client_compile(&comp_args) != NULL ||
      comp_args.executable == NULL) {
    return 6;
  }

  PJRT_XlaTransform_Unregister_Args xt_unreg = {
      sizeof(PJRT_XlaTransform_Unregister_Args), NULL, "c11_test_pass", 13U};
  if (PJRT_XlaTransform_Unregister(&xt_unreg) != NULL) {
    return 53;
  }

  /* PJRT_AbiVersion_IsCompatible: Runtime <-> Runtime and Runtime <->
   * Executable */
  PJRT_LoadedExecutable_GetExecutable_Args get_ex = {
      sizeof(PJRT_LoadedExecutable_GetExecutable_Args), NULL,
      comp_args.executable, NULL};
  if (api->pjrt_loaded_executable_get_executable(&get_ex) != NULL ||
      get_ex.executable == NULL) {
    return 54;
  }
  PJRT_AbiVersion_Get_Args rt_abi_args = {sizeof(PJRT_AbiVersion_Get_Args),
                                          NULL, create_args.client, NULL};
  PJRT_AbiVersion_Get_Args ex_abi_args = {sizeof(PJRT_AbiVersion_Get_Args),
                                          NULL, get_ex.executable, NULL};
  if (PJRT_AbiVersion_ClientRuntime(&rt_abi_args) != NULL ||
      PJRT_AbiVersion_ExecutableGet(&ex_abi_args) != NULL ||
      rt_abi_args.abi_version_out == NULL ||
      ex_abi_args.abi_version_out == NULL) {
    return 55;
  }
  PJRT_AbiVersion_IsCompatible_Args abi_cmp = {
      sizeof(PJRT_AbiVersion_IsCompatible_Args), NULL,
      rt_abi_args.abi_version_out, ex_abi_args.abi_version_out, 0U};
  if (PJRT_AbiVersion_RuntimeCompatibleWithExecutable(&abi_cmp) != NULL ||
      abi_cmp.is_compatible_out != 1U) {
    return 56;
  }
  abi_cmp.other_abi_version = rt_abi_args.abi_version_out;
  abi_cmp.is_compatible_out = 0U;
  if (PJRT_AbiVersion_RuntimeCompatibleWithRuntime(&abi_cmp) != NULL ||
      abi_cmp.is_compatible_out != 1U) {
    return 57;
  }
  PJRT_AbiVersion_Destroy_Args rt_abi_del = {
      sizeof(PJRT_AbiVersion_Destroy_Args), NULL, rt_abi_args.abi_version_out};
  PJRT_AbiVersion_Destroy_Args ex_abi_del = {
      sizeof(PJRT_AbiVersion_Destroy_Args), NULL, ex_abi_args.abi_version_out};
  if (PJRT_AbiVersion_RuntimeDestroy(&rt_abi_del) != NULL ||
      PJRT_AbiVersion_ExecutableDestroy(&ex_abi_del) != NULL) {
    return 58;
  }
  PJRT_Executable_Destroy_Args raw_ex_del = {
      sizeof(PJRT_Executable_Destroy_Args), NULL, get_ex.executable};
  if (api->pjrt_executable_destroy(&raw_ex_del) != NULL) {
    return 59;
  }

  /* PJRT_RawBuffer_* lifecycle, offset H2D/D2H, and out-of-bounds error
   * contracts */
  PJRT_Client_Devices_Args devs_args = {sizeof(PJRT_Client_Devices_Args), NULL,
                                        create_args.client, NULL, 0U};
  if (api->pjrt_client_addressable_devices(&devs_args) != NULL ||
      devs_args.num_devices == 0U) {
    return 60;
  }
  const float init_vals[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  const int64_t rb_dims[1] = {4};
  PJRT_Client_BufferFromHostBuffer_Args h2d = {
      sizeof(PJRT_Client_BufferFromHostBuffer_Args),
      NULL,
      create_args.client,
      init_vals,
      11,
      0,
      rb_dims,
      1U,
      NULL,
      0U,
      0,
      0,
      devs_args.devices[0],
      NULL,
      NULL,
      NULL,
      NULL};
  if (api->pjrt_client_buffer_from_host_buffer(&h2d) != NULL ||
      h2d.buffer == NULL) {
    return 61;
  }
  PJRT_Event_Destroy_Args ev_del = {sizeof(PJRT_Event_Destroy_Args), NULL,
                                    h2d.done_with_host_buffer};
  (void)api->pjrt_event_destroy(&ev_del);

  PJRT_RawBuffer_Create_Args rb_create = {sizeof(PJRT_RawBuffer_Create_Args),
                                          NULL, h2d.buffer, NULL};
  if (PJRT_RawBuffer_CreateRawAliasOfBuffer(&rb_create) != NULL ||
      rb_create.raw_buffer_out == NULL) {
    return 62;
  }
  PJRT_RawBuffer_GetHostPointer_Args rb_ptr = {
      sizeof(PJRT_RawBuffer_GetHostPointer_Args), NULL,
      rb_create.raw_buffer_out, NULL};
  PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args rb_sz = {
      sizeof(PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args), NULL,
      rb_create.raw_buffer_out, 0U};
  PJRT_RawBuffer_GetMemorySpace_Args rb_mem = {
      sizeof(PJRT_RawBuffer_GetMemorySpace_Args), NULL,
      rb_create.raw_buffer_out, NULL};
  if (PJRT_RawBuffer_GetHostPointer(&rb_ptr) != NULL ||
      rb_ptr.host_pointer_out == NULL ||
      PJRT_RawBuffer_GetOnDeviceSizeInBytes(&rb_sz) != NULL ||
      rb_sz.on_device_size_in_bytes != 256U ||
      PJRT_RawBuffer_GetMemorySpace(&rb_mem) != NULL ||
      rb_mem.memory_space_out == NULL) {
    return 63;
  }
  const float patch_vals[2] = {99.0F, 100.0F};
  PJRT_RawBuffer_CopyRaw_Args rb_h2d = {sizeof(PJRT_RawBuffer_CopyRaw_Args),
                                        NULL,
                                        rb_create.raw_buffer_out,
                                        (void *)(uintptr_t)patch_vals,
                                        4,
                                        8,
                                        NULL};
  if (PJRT_RawBuffer_CopyRawHostToDevice(&rb_h2d) != NULL) {
    return 64;
  }
  ev_del.event = rb_h2d.event;
  (void)api->pjrt_event_destroy(&ev_del);

  float readback[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  PJRT_RawBuffer_CopyRaw_Args rb_d2h = {sizeof(PJRT_RawBuffer_CopyRaw_Args),
                                        NULL,
                                        rb_create.raw_buffer_out,
                                        readback,
                                        0,
                                        16,
                                        NULL};
  if (PJRT_RawBuffer_CopyRawDeviceToHost(&rb_d2h) != NULL ||
      readback[0] != 1.0F || readback[1] != 99.0F || readback[2] != 100.0F ||
      readback[3] != 4.0F) {
    return 65;
  }
  ev_del.event = rb_d2h.event;
  (void)api->pjrt_event_destroy(&ev_del);

  /* Out-of-bounds offset + transfer_size must return PJRT_Error* */
  PJRT_RawBuffer_CopyRaw_Args rb_oob = {sizeof(PJRT_RawBuffer_CopyRaw_Args),
                                        NULL,
                                        rb_create.raw_buffer_out,
                                        readback,
                                        252,
                                        8,
                                        NULL};
  PJRT_Error *oob_err = PJRT_RawBuffer_CopyRawDeviceToHost(&rb_oob);
  if (oob_err == NULL) {
    return 66;
  }
  PJRT_Error_Destroy_Args oob_del = {sizeof(PJRT_Error_Destroy_Args), NULL,
                                     oob_err};
  api->pjrt_error_destroy(&oob_del);

  PJRT_RawBuffer_Destroy_Args rb_del = {sizeof(PJRT_RawBuffer_Destroy_Args),
                                        NULL, rb_create.raw_buffer_out};
  if (PJRT_RawBuffer_Destroy(&rb_del) != NULL) {
    return 67;
  }
  PJRT_Buffer_Destroy_Args b_del = {sizeof(PJRT_Buffer_Destroy_Args), NULL,
                                    h2d.buffer};
  if (api->pjrt_buffer_destroy(&b_del) != NULL) {
    return 68;
  }

  /* PJRT_MultiSlice_Config_* lifecycle */
  PJRT_Megascale_CreateDefaultConfig_Args msc_cr = {
      sizeof(PJRT_Megascale_CreateDefaultConfig_Args), NULL, NULL, NULL};
  if (PJRT_Megascale_CreateDefaultConfig(&msc_cr) != NULL ||
      msc_cr.config_out == NULL) {
    return 69;
  }
  PJRT_MultiSlice_Config_NumSlices_Args msc_ns = {
      sizeof(PJRT_MultiSlice_Config_NumSlices_Args), NULL, msc_cr.config_out,
      0U};
  PJRT_MultiSlice_Config_NumDevicesPerSlice_Args msc_nd = {
      sizeof(PJRT_MultiSlice_Config_NumDevicesPerSlice_Args), NULL,
      msc_cr.config_out, 0U, 0U};
  PJRT_MultiSlice_Config_SliceId_Args msc_sid = {
      sizeof(PJRT_MultiSlice_Config_SliceId_Args), NULL, msc_cr.config_out,
      99U};
  PJRT_MultiSlice_Config_Serialize_Args msc_ser = {
      sizeof(PJRT_MultiSlice_Config_Serialize_Args),
      NULL,
      msc_cr.config_out,
      NULL,
      0U,
      NULL};
  if (PJRT_MultiSlice_Config_NumSlices(&msc_ns) != NULL ||
      msc_ns.num_slices < 1U ||
      PJRT_MultiSlice_Config_NumDevicesPerSlice(&msc_nd) != NULL ||
      msc_nd.num_devices_per_slice_map < 1U ||
      PJRT_MultiSlice_Config_SliceId(&msc_sid) != NULL ||
      PJRT_MultiSlice_Config_Serialize(&msc_ser) != NULL ||
      msc_ser.serialized_bytes_size == 0U) {
    return 70;
  }
  PJRT_Metadata_Destroy_Args msc_ser_del = {sizeof(PJRT_Metadata_Destroy_Args),
                                            NULL, msc_ser.serialized_config};
  (void)PJRT_ExecutableMetadata_Destroy(&msc_ser_del);
  PJRT_MultiSlice_Config_Destroy_Args msc_del = {
      sizeof(PJRT_MultiSlice_Config_Destroy_Args), NULL, msc_cr.config_out};
  if (PJRT_MultiSlice_Config_Destroy(&msc_del) != NULL) {
    return 71;
  }

  /* TpuEmbeddingEngine_ConfigureMemory + CollateMemory with 0x00/0xFF binary
   * bytes */
  const uint8_t bin_cfg_in[6] = {0x00U, 0xFFU, 0x10U, 0x00U, 0xFEU, 0x01U};
  size_t mem0_sz = 0U;
  char *mem0_ptr = NULL;
  TpuStatus *emb_st = TpuStatus_New();
  TpuEmbeddingEngine_ConfigureMemory_Params cfg_mem = {
      sizeof(TpuEmbeddingEngine_ConfigureMemory_Params),
      NULL,
      2,
      0,
      (const char *)bin_cfg_in,
      sizeof(bin_cfg_in),
      &mem0_sz,
      &mem0_ptr,
      emb_st};
  if (TpuEmbeddingEngine_ConfigureMemory(&cfg_mem) != 0 ||
      !TpuStatus_Ok(emb_st) || mem0_ptr == NULL || mem0_sz == 0U) {
    return 72;
  }
  TpuSerializedProto task_slices[2] = {
      {mem0_ptr, mem0_sz},
      {mem0_ptr, mem0_sz},
  };
  size_t merged_sz = 0U;
  char *merged_ptr = NULL;
  TpuEmbeddingEngine_CollateMemory_Params col_mem = {
      sizeof(TpuEmbeddingEngine_CollateMemory_Params),
      NULL,
      2U,
      task_slices,
      &merged_sz,
      &merged_ptr,
      emb_st};
  if (TpuEmbeddingEngine_CollateMemory(&col_mem) != 0 ||
      !TpuStatus_Ok(emb_st) || merged_ptr == NULL || merged_sz == 0U) {
    return 73;
  }
  TpuConfigurationApi_FreeCharArray(mem0_ptr);
  TpuConfigurationApi_FreeCharArray(merged_ptr);
  TpuStatus_Free(emb_st);

  PJRT_LoadedExecutable_Destroy_Args del_exec;
  memset(&del_exec, 0, sizeof(del_exec));
  del_exec.struct_size = sizeof(del_exec);
  del_exec.executable = comp_args.executable;
  if (api->pjrt_loaded_executable_destroy(&del_exec) != NULL) {
    return 7;
  }

  PJRT_Client_Destroy_Args del_cli;
  memset(&del_cli, 0, sizeof(del_cli));
  del_cli.struct_size = sizeof(del_cli);
  del_cli.client = create_args.client;
  if (api->pjrt_client_destroy(&del_cli) != NULL) {
    return 8;
  }

  /* TpuDriver: GetCompiledProgramShape + FreeCompiledProgramShape +
   * Load/UnloadProgram (no double-free) */
  TpuDriver *drv = TpuDriver_Open("local://0");
  if (drv == NULL) {
    return 9;
  }
  TpuCompiledProgramHandle *cprog =
      TpuDriver_CompileProgramFromText(drv, mlir_src, 1, 0, NULL);
  TpuCompiledProgramShape *cshape =
      TpuDriver_GetCompiledProgramShape(drv, cprog);
  TpuLoadedProgramHandle *lprog = TpuDriver_LoadProgram(drv, 0, cprog, 0, NULL);
  if (cprog == NULL || cshape == NULL || lprog == NULL) {
    return 91;
  }
  TpuDriver_FreeCompiledProgramShape(cshape);
  TpuEvent *unld_ev = TpuDriver_UnloadProgram(drv, lprog, 0, NULL);
  TpuDriver_FreeEvent(unld_ev);
  TpuDriver_FreeCompiledProgramHandle(cprog);
  TpuStatus *rst_st = TpuDriver_Reset(drv);
  TpuStatus_Free(rst_st);
  TpuDriver_Close(drv);

  PJRT_Client_Create_Args doc_create_args = {
      sizeof(PJRT_Client_Create_Args),
      NULL,
      NULL,
      0,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
  };
  if (api->pjrt_client_create(&doc_create_args) != NULL) {
    return 10;
  }
  PJRT_Client_Destroy_Args doc_destroy_args = {
      sizeof(PJRT_Client_Destroy_Args),
      NULL,
      doc_create_args.client,
  };
  if (api->pjrt_client_destroy(&doc_destroy_args) != NULL) {
    return 11;
  }

  PJRT_Triton_FreeAsm(NULL, 0U);
  return 0;
}
