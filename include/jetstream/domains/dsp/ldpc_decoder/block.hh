#ifndef JETSTREAM_DOMAINS_DSP_LDPC_DECODER_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_LDPC_DECODER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct LdpcDecoder : public Block::Config {
    U64 iterations = 50;

    JST_BLOCK_TYPE(ldpc_decoder);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(iterations);
    JST_BLOCK_DESCRIPTION(
        "LDPC Decoder",
        "CCSDS AR4JA rate-1/2 k=1024 LDPC decoder (flooded min-sum).",
        "# LDPC Decoder\n"
        "\n"
        "Decodes log-likelihood ratios from the CCSDS AR4JA rate-1/2, k=1024 code "
        "(n=2560, 2048 transmitted symbols, 512 punctured) with a flooded min-sum "
        "iteration and early parity termination.\n"
        "\n"
        "- **One frame per lane.** The last axis must be exactly 2048 LLRs; every "
        "other axis is decoded independently.\n"
        "- **Output is the full 2560-bit codeword.** Hard decisions as F32 0/1; the "
        "last 1024 entries are the information bits.\n"
        "- **Positive LLR = bit 0.** Matches BPSK 0 = +1 convention.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 | Last axis exactly 2048 channel LLRs. |\n"
        "| **Output** (out) | F32 | Input shape with the last axis replaced by "
        "2560 hard decisions. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Iterations** | 50 | 1 to 200 | Maximum min-sum iterations before "
        "giving up; decoding stops earlier when all parity checks pass. |\n");
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_LDPC_DECODER_BLOCK_HH
