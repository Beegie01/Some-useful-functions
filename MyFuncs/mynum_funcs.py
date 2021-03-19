def float_range(start, stop, step):
    '''
    returns a list of floating point numbers

    Example:
    float_range(0, 1, 0.1) -->

    Output:
    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    '''

    first = 'first'
    nums = []
    count = None

    # to stop an infinite loop condition
    if step < 0:
        return "Error: negative step not allowed!"

    while count == None or count <= round(stop, 1):
        if first != 'done':
            count = start
            nums.append(round(count, 1))
            first = 'done'
            continue

        count += step

        if count <= stop:
            nums.append(round(count, 1))

    return nums
