def hashAttributes(attributes):
    res = 0
    for attribute in attributes:
        res = res + (1 << attribute)
    return res
