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

def round_norm(num: float=0, dec_point: int=0):
    '''
    NOTE: Only works with decimals, use the in-built round or numpy.round
    to round whole numbers
    to round up or round down a number regardless of the oddity
    rounds up the number to the decimal point if preceding value is 5 or greater
    rounds down the number to the decimal point if the preceding value is less than 5
    '''
    if dec_point < 0:
        raise ValueError('You supplied a negative decimal point\n\
        Please use np.round or round for such operations')

    if '.' not in str(num):
        raise ValueError('Cannot perform operation on a whole number\n\
        Please enter a decimal number')
    #  when only zeros present after the decimal point
    if float(num) == int(num):
        return int(num)

    dd, dd['whole'], dd['decimal'] = {}, str(num).split('.')[0], str(num).split('.')[-1]
    #  dd['rev_dec'] = decimal[::-1]

    #  when no need to round
    if dec_point == len(dd['decimal']):
        return num

    #  when decimal point is greater than the number of figures
    #  decimal point, simply fill the remaining space with zeros
    if dec_point > len(dd['decimal']):
        return f"{dd['whole']}.{dd['decimal'][::-1].zfill(dec_point)[::-1]}"

    if dec_point == 0:  # if decimal point is 0:
        #  when first figure after the decimal point is 5 or greater,
        #  round up the preceding figure
        if int(dd['decimal'][dec_point]) >= 5:
            return int(f"{dd['whole'][:-1]}{int(dd['whole'][-1])+1}")
        #  when first decimal number is less than 5, round down
        else:
            return int(dd['whole'])

    else:  #  if decimal point is not zero
        #  when the figure at the decimal point >= 5
        if dec_point == 1:
            if int(dd['decimal'][dec_point]) >= 5:  #  round up
                return float(f"{dd['whole']}.{int(dd['decimal'][dec_point-1])+1}")
            else:  #  round down
                return float(f"{dd['whole']}.{dd['decimal'][dec_point-1]}")

        if dec_point > 1:
             if int(dd['decimal'][dec_point]) >= 5:  #  round up
                return float( f"{dd['whole']}.{dd['decimal'][:dec_point-1]}{int(dd['decimal'][dec_point-1])+1}" )
             else:  #  round down
                return float(f"{dd['whole']}.{dd['decimal'][:dec_point]}")
