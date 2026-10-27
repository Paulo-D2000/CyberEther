#ifndef JETSTREAM_DOMAINS_DSP_LDPC_DECODER_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_LDPC_DECODER_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct LdpcDecoder : public Module::Config {
    U64 iterations = 50;

    JST_MODULE_TYPE(ldpc_decoder);
    JST_MODULE_PARAMS(iterations);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_LDPC_DECODER_MODULE_HH
