def remove_module_prefix(state_dict, prefix="module."):
    """
    state_dict의 key에서 prefix를 제거한 새로운 OrderedDict 반환
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            name = k[len(prefix):]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict