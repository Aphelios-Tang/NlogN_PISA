EMBED_DIM_BASE = 256


def pad_embed_dim(embed_dim):
    return (embed_dim + EMBED_DIM_BASE - 1) // EMBED_DIM_BASE * EMBED_DIM_BASE


def compute_mid_dim(embed_dim, c=8 / 3):
    return pad_embed_dim(int(embed_dim * c))


def print_array(array):
    string = array[0]
    for col in array[1:]:
        string += f",{col}"

    print(string)


def print_dict(array, dict):
    string = f"{dict[array[0]]}"
    for col in array[1:]:
        string += f",{dict[col]}"

    print(string)


def get_params(
    token_mixer,
    channel_mixer,
    num_layer,
    embed_dim,
    head_dim,
    original_vocab_size,
    q_rank=8,
    kv_rank=2,
    num_heads=-1,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    mfa_head_dim=128,
    kv_heads=-1,
):
    num_heads = embed_dim // head_dim if num_heads == -1 else num_heads
    kv_heads = num_heads if kv_heads == -1 else kv_heads
    vocab_size = pad_embed_dim(original_vocab_size)
    if token_mixer == "hgru3":
        token_mixer_params = (
            embed_dim * embed_dim * 4 + embed_dim * num_heads + 2 * embed_dim * head_dim
        )
    elif token_mixer in ["hgru2", "attn"]:
        token_mixer_params = embed_dim * embed_dim * 4
    elif token_mixer == "mpa":
        mid_dim = head_dim * num_heads
        token_mixer_params = embed_dim * (mid_dim + head_dim + num_heads) * 2
    elif token_mixer == "mpa-q-16":
        mid_dim = head_dim * num_heads
        token_mixer_params = embed_dim * (
            mid_dim + (head_dim + num_heads) * 2
        ) + embed_dim * q_rank * (head_dim + num_heads)
    elif token_mixer in ["tpa", "tpa-kv1"]:
        mid_dim = head_dim * num_heads
        token_mixer_params = embed_dim * (
            mid_dim + (q_rank + 2 * kv_rank) * (head_dim + num_heads)
        )
    elif token_mixer == "mla":
        qk_nope_head_dim = head_dim
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = head_dim
        token_mixer_params = (
            embed_dim * q_lora_rank
            + q_lora_rank * num_heads * q_head_dim
            + embed_dim * (kv_lora_rank + qk_rope_head_dim)
            + kv_lora_rank * num_heads * (qk_nope_head_dim + v_head_dim)
            + num_heads * v_head_dim * embed_dim
        )
        a1 = q_lora_rank
        a2 = num_heads * q_head_dim
        a3 = kv_lora_rank + qk_rope_head_dim
        a4 = num_heads * (qk_nope_head_dim + v_head_dim)
        a5 = num_heads * v_head_dim
    elif token_mixer == "mfa":
        mid_dim = mfa_head_dim * num_heads
        token_mixer_params = (
            embed_dim * (3 + num_heads) * mfa_head_dim
            + num_heads * mfa_head_dim * mfa_head_dim
        )
    elif token_mixer == "mfa-share":
        mid_dim = mfa_head_dim * num_heads
        token_mixer_params = (
            embed_dim * (2 + num_heads) * mfa_head_dim
            + (num_heads + 1) * mfa_head_dim * mfa_head_dim
            + num_heads * mfa_head_dim
        )
    elif token_mixer in ["opa", "opa_v2"]:
        mid_dim = head_dim * num_heads
        if head_dim <= 64:
            d = 8
        else:
            d = 16
        token_mixer_params = embed_dim * (
            2 * mid_dim + 2 * num_heads * (d + head_dim // d)
        )
    elif token_mixer == "nsa":
        mid_dim = head_dim * num_heads
        kv_dim = head_dim * kv_heads
        token_mixer_params = 2 * embed_dim * (mid_dim + kv_dim)

    if channel_mixer == "ffn":
        coef = 2
        mid_dim = compute_mid_dim(embed_dim, 4)
    elif channel_mixer == "glu":
        coef = 3
        mid_dim = compute_mid_dim(embed_dim, 8 / 3)
    channel_mixer_params = mid_dim * embed_dim * coef
    non_embed_params = round(
        num_layer * (token_mixer_params + channel_mixer_params) / 1e9, 4
    )  # B
    embed_params = round(embed_dim * vocab_size / 1e9, 4)  # B
    total_params_tied = round(non_embed_params + embed_params, 4)
    total_params_untied = round(non_embed_params + embed_params * 2, 4)

    res = {
        "token_mixer": token_mixer,
        "channel_mixer": channel_mixer,
        "num_layer": num_layer,
        "embed_dim": embed_dim,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "mid_dim": mid_dim,
        "original_vocab_size": original_vocab_size,
        "vocab_size": vocab_size,
        "non_embed_params": non_embed_params,
        "embed_params": embed_params,
        "total_params_tied": total_params_tied,
        "total_params_untied": total_params_untied,
        "q_rank": q_rank,
        "kv_rank": kv_rank,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "mfa_head_dim": mfa_head_dim,
        "kv_heads": kv_heads,
    }
    return res


original_vocab_size = 50272

cols = [
    "token_mixer",
    "channel_mixer",
    "non_embed_params",
    "embed_params",
    "total_params_tied",
    "total_params_untied",
    "token_mixer",
    "channel_mixer",
    "num_layer",
    "embed_dim",
    "head_dim",
    "num_heads",
    "mid_dim",
    "original_vocab_size",
    "vocab_size",
    "q_rank",
    "kv_rank",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "mfa_head_dim",
    "kv_heads",
]

hyper_params = {
    "160m": {
        "num_layer": 12,
        "embed_dim": 768,
        "head_dim": 64,
        "q_lora_rank": 512,
        "kv_lora_rank": 256,
        "qk_rope_head_dim": 64,
        "mfa_head_dim": 128,
    },
    "310m": {
        "num_layer": 24,
        "embed_dim": 1024,
        "head_dim": 64,
        "q_lora_rank": 512,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "mfa_head_dim": 128,
    },
    "1.5B": {
        "num_layer": 24,
        "embed_dim": 2048,
        "head_dim": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 384,
        "qk_rope_head_dim": 128,
        "mfa_head_dim": 256,
    },
    "3B": {
        "num_layer": 32,
        "embed_dim": 2560,
        "head_dim": 128,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 128,
        "mfa_head_dim": 256,
    },
    "7B": {
        "num_layer": 32,
        "embed_dim": 4096,
        "head_dim": 128,
        "q_lora_rank": 3584,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 128,
        "mfa_head_dim": 256,
    },
    "13B": {
        "num_layer": 40,
        "embed_dim": 5120,
        "head_dim": 128,
        "q_lora_rank": 4608,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 128,
        "mfa_head_dim": 256,
    },
}

print_array(cols)

token_mixers = [
    "attn",
    "mpa",
    "mpa-q-16",
    "tpa",
    "tpa-kv1",
    "mla",
    "mfa",
    "mfa-share",
    "opa",
    "opa_v2",
    "nsa",
]
channel_mixers = ["glu"]

for token_mixer in token_mixers:
    for channel_mixer in channel_mixers:
        for key in hyper_params.keys():
            value = hyper_params[key]
            num_layer = value["num_layer"]
            embed_dim = value["embed_dim"]
            head_dim = value["head_dim"]
            q_lora_rank = value["q_lora_rank"]
            kv_lora_rank = value["kv_lora_rank"]
            qk_rope_head_dim = value["qk_rope_head_dim"]
            mfa_head_dim = value["mfa_head_dim"]
            kv_heads = -1
            # for tpa
            if token_mixer in ["tpa", "tpa-kv1"]:
                num_heads = embed_dim // head_dim * 3
            elif token_mixer in [
                "mpa",
            ]:
                num_heads = embed_dim // head_dim * 2
            elif token_mixer == "mfa":
                if key in ["13B"]:
                    num_heads = int(embed_dim // mfa_head_dim * 3.75)
                elif key in ["7B"]:
                    num_heads = int(embed_dim // mfa_head_dim * 3.5)
                else:
                    num_heads = int(embed_dim // mfa_head_dim * 3.25)
            elif token_mixer == "mfa-share":
                if key in ["13B"]:
                    num_heads = int(embed_dim // mfa_head_dim * 3.75)
                elif key in ["7B"]:
                    num_heads = int(embed_dim // mfa_head_dim * 3.5)
                else:
                    num_heads = int(embed_dim // mfa_head_dim * 3.25)
            elif token_mixer == "mpa-q-16":
                if key in ["7B", "13B"]:
                    num_heads = int(embed_dim // head_dim * 3)
                else:
                    num_heads = int(embed_dim // head_dim * 2.5)
            elif token_mixer == "opa":
                num_heads = int(embed_dim // head_dim * 1.5)
            elif token_mixer == "opa_v2":
                head_dim = 256
                num_heads = int(embed_dim // head_dim * 1.5)
            elif token_mixer == "nsa":
                if key in [
                    "7B",
                ]:
                    num_heads = 64
                    kv_heads = 2
                elif key in ["13B"]:
                    num_heads = 64
                    kv_heads = 4
                else:
                    num_heads = 32
                    kv_heads = 2
            else:
                num_heads = -1

            if token_mixer in ["tpa", "tpa-kv1"]:
                if token_mixer == "tpa-kv1":
                    kv_rank = 1
                else:
                    kv_rank = 2
                if key in ["7B", "13B"]:
                    q_rank = 16
                else:
                    q_rank = 8
            else:
                if token_mixer == "mpa-q-16":
                    q_rank = 16
                else:
                    q_rank = -1
                kv_rank = -1

            res = get_params(
                token_mixer=token_mixer,
                channel_mixer=channel_mixer,
                num_layer=num_layer,
                embed_dim=embed_dim,
                head_dim=head_dim,
                original_vocab_size=original_vocab_size,
                num_heads=num_heads,
                q_rank=q_rank,
                kv_rank=kv_rank,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                mfa_head_dim=mfa_head_dim,
                kv_heads=kv_heads,
            )
            res["token_mixer"] = token_mixer
            res["channel_mixer"] = channel_mixer
            print_dict(cols, res)
