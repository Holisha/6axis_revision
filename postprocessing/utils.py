from argparse import ArgumentParser

def get_len(data):
    """get the original length of the stroke

    Args:
        data (pandas.Dataframe): the stroke data to count the length

    Returns:
        int: the original length of the stroke
    """
    match = data.iloc[:, :-1].eq(data.iloc[:, :-1].shift())

    # get the length of the different rows
    stroke_len = len(
        match.groupby(
            match.iloc[:, 0]
        ).groups.get(False)
    )
    return stroke_len

def argument_setting():
    r"""
    return arguments
    """
    parser = ArgumentParser()

    parser.add_argument('--path', type=str, default='./output',
                        help='set the data path (default: ./output)')

    return parser.parse_args()