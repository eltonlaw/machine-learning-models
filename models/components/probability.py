""" Bunch of random statistics stuff"""

# pylint: disable=invalid-name
def _factorial(n):
    """ n*n-1*n-2...2*1 """
    if n < 1:
        return 1
    else:
        return n * _factorial(n-1)


def _binomial_coefficient(n, k):
    """ n choose k """
    return _factorial(n)/(_factorial(n-k)*_factorial(k))


def sample_num_events(n, k, replace=True, order_matters=True):
    """ Return number of events dependent on the existence of replacement
    and whether order matters

    Parameters
    ----------
    n: int
        Number of possible outcomes
    k: int
        Number of events

    Returns
    -------
    num_events
        The number of possible outcomes

    """
    if replace and order_matters:
        num_events = n**k
    elif replace and not order_matters:
        num_events = _binomial_coefficient(n+k-1, k)
    elif not replace and order_matters:
        num_events = 1.
        for i in range(k):
            num_events *= (n-i)
    elif not replace and not order_matters:
        num_events = _binomial_coefficient(n, k)
    return num_events


def two_same_birthdays(n):
    """ Calculate the probability of two people in a room having the same
    birthday with `n` people in the room. Assumes that every birthday is equal

    Each extra person takes up one slot on the calendar...
    Imagine a line of people coming into a room one at a time, each time
    someone enters the room they take another slot on the calendar. We
    count the number of scenarios where each time someone enters, they
    take a slot on the calendar that hasn't been taken yet. That's what
    the `numerator` calculated below represents.

    We then divide by the total number of all birthday combinations.
    """
    if n > 365:
        pr_match = 1
    else:
        numerator = 1.
        for i in range(n):
            numerator *= (365.-i)
        pr_nomatch = numerator/(365.**n)
        pr_match = 1 - pr_nomatch
    return pr_match

if __name__ == "__main__":
    print(_factorial(10))
    # 3628800
    print(_binomial_coefficient(52, 5))
    # 2598960
    print(_binomial_coefficient(10, 6) == _binomial_coefficient(10, 4))
    # True // (n choose k) is equivalent to (n choose n-k)
    print(sample_num_events(8, 3, replace=True, order_matters=True))
    # 512
    print(sample_num_events(8, 3, replace=False, order_matters=True))
    # 336
    print(sample_num_events(8, 3, replace=True, order_matters=False))
    # 120
    print(sample_num_events(8, 3, replace=False, order_matters=False))
    # 56
    print(two_same_birthdays(50))
    # 0.9703735795779884
    print(two_same_birthdays(23))
    # 0.5072972343239854
