from functools import partial

features = None


def set_features(feats):
    global features
    features = feats


#should not pass line_features to out
def has_keyword(keyword, out1, out2, line_features):
    if (int(line_features[features.index(keyword)])) > 0:
        return out1(line_features)
    else:
        return out2(line_features)


def count(_):
    return True


def no_count(_):
    return False


def primitive_feature(keyword):
    return partial(primitive_keyword, keyword)


def primitive_keyword(keyword, out1, out2):
    return partial(has_keyword, keyword, out1, out2)


