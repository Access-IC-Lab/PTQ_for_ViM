class SimilarityQuantConfig:
    def __init__(self, metric=None, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10):
        self.metric = metric
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n


class KScaledQuantConfig:
    def __init__(
        self,
        k_scaled=False,
        k_scaled_mode="channel_wise",  # "channel_wise" or "token_wise"
        k=2,
        k_scaled_power_of_two_scaling=False,
        k_scaled_clusters=None,
    ):
        self.k_scaled = k_scaled
        self.k_scaled_mode = k_scaled_mode
        self.k = k
        self.k_scaled_power_of_two_scaling = k_scaled_power_of_two_scaling
        self.k_scaled_clusters = k_scaled_clusters


class QuantConfig:
    def __init__(
        self,
        quant=False,
        qmode="minmax",  # "minmax" or "quantile" or "similarity"
        bin="uniform",  # "uniform" or "power_of_two"
        bit=8,
        quantile=0.9999,
        symmetric=True,
        similarity_config=SimilarityQuantConfig(),
        k_scaled_config=KScaledQuantConfig(),
    ):
        self.quant = quant
        self.qmode = qmode
        self.bin = bin
        self.bit = bit
        self.quantile = quantile
        self.symmetric = symmetric
        self.similarity_config = similarity_config
        self.k_scaled_config = k_scaled_config
        self.interval = None
        self.zeropoint = 0
        self.qmax = 2 ** (self.bit - 1)
