from factorisation.densenet import get_increasing_grouped_densenet121, get_increasing_mix_bottlenecks_densenet121

model_functions = {
    "grouped1": get_increasing_grouped_densenet121,
    "grouped2": get_increasing_mix_bottlenecks_densenet121,
}