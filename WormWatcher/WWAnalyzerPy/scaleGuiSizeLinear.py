"""
    Shortcut function for linear of a gui element size

        EXAMPLE: x is user screen height and y is height of scrolltext in lines.

        x = User actual screen height
        xmin = smallest allowed screen height
        ymin = value of height in lines at the smallest screen height
        xmax = largest allowed screen height
        ymax = vakye if height in lines at the largest screen height

        Screen sizes smaller than xmin will use ymin as the number of lines
"""


def scaleGuiSizeLinear(x,xmin,ymin,xmax,ymax,output_int = True):

    # Return min or max value if screen size is small or large
    if x <= xmin:
        return ymin

    elif x >= xmax:
        return ymax

    # Otherwise, Calculate slope from 2 points
    m = (ymax - ymin) / (xmax - xmin)

    # Calculate intercept for y = mx + b
    b = ymax - m * xmax

    # Return intermediate size
    y = m*x + b
    if output_int:
        y = int(y)

    return y

    
