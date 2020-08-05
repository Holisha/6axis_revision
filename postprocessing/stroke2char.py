import pandas as pd

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


def stroke2char(target_data, input_data, output_data, test_target, test_input, test_output, dir_path, stroke_len, index):
    """Merge each stroke into a single character

    Args:
        target_data (pandas.DataFrame): original target data
        input_data (pandas.DataFrame): original input data
        output_data (pandas.DataFrame): original output data
        test_target (pandas.DataFrame): all test target data
        test_input (pandas.DataFrame): all test input data
        test_output (pandas.DataFrame): all test output data
        dir_path (string): the path of directory
        stroke_len (int): the length of stroke
        index (string): the stroke index

    Returns:
        Boolean, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame: True, test_target, test_input, test_output
    """

    # Update stroke number
    target_data[6] = [f'stroke{index}'] * stroke_len
    input_data[6] = [f'stroke{index}'] * stroke_len
    output_data[6] = [f'stroke{index}'] * stroke_len

    # append data
    test_target = test_target.append(target_data, ignore_index=True)
    test_input = test_input.append(input_data, ignore_index=True)
    test_output = test_output.append(output_data, ignore_index=True)

    test_target.to_csv(f'{dir_path}/test_all_target.csv', header=False, index=False)
    test_input.to_csv(f'{dir_path}/test_all_input.csv', header=False, index=False)
    test_output.to_csv(f'{dir_path}/test_all_output.csv', header=False ,index=False)

    return True, test_target, test_input, test_output