#ifndef JETSTREAM_DOMAINS_DSP_LDPC_DECODER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_LDPC_DECODER_MODULE_IMPL_HH

#include <jetstream/domains/dsp/ldpc_decoder/module.hh>
#include <jetstream/detail/module_impl.hh>

#include "ldpc_ar4ja_tables.hh"

#include <vector>

namespace Jetstream::Modules {

using namespace ::ldpc;

struct LdpcDecoderImpl : public Module::Impl, public DynamicConfig<LdpcDecoder> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
    Index sampleAxis = 0;
    U64 laneCount = 0;
    std::vector<F32> llr;
    std::vector<F32> checkToVar;
    std::vector<F32> varSum;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_LDPC_DECODER_MODULE_IMPL_HH
