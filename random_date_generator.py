import random
import datetime


class RandomDate(object):
    
    def __init__(self):
        print("RandomDate class imported!")
    
    def random_year(self):
        """
        randomly generate an integer value for year ranging
        from last year (current year - 1)
        to 120 years before (current year - 120)
        """
        return random.randint(datetime.date.today().year - 120, datetime.date.today().year - 1)
    
    def leap_year_checker(self, year_inp):
        """
        checks for the occurrence of a leap year
        a leap year occurs when year is not divisible by 100,
        but is divisible by 4
        or when year is divisible by 400
        """
        is_leap_year = False
        # check if the year argument is a leap year
        if (year_inp % 100) and not (year_inp % 4):
            is_leap_year = True
        elif not (year_inp % 400):
            is_leap_year = True

        return is_leap_year

    def random_month(self) -> int:
        """
        randomly generate a month integer ranging from 1-12 inclusive
        """
        return random.randint(1, 12)
    
    def random_day(self, max_days: int) -> int:
        """
        randomly output an integer from range 1 - max_days
        """
        return random.randint(1, max_days)
    
    def check_max_days(self, year_inp, month: int) -> int:
        """
        generate corresponding maximum number of days
        based on the month argument
        """
        odd_months = {
            2: (28, 29),
            (4, 6, 9, 11): 30
        }
        d = 31
        # get the maximum number of days for the corresponding month argument
        for mon, days in odd_months.items():
            if isinstance(mon, tuple):
                if month in mon:  # if given month belongs to the grouped key category
                    d = days
                    # print(f"Max days is {d}")
            else:
                if (month == mon) and self.leap_year_checker(year_inp):  # if given month is february and year is leap year
                    d = days[1]
                    # print(f"Max days is {d}")
                # if given month is february and year is not leap year
                elif (month == mon) and not self.leap_year_checker(year_inp):
                    d = days[0]
                    # print(f"Max days is {d}")
        return d
    
    def generate_random_date(self) -> str:
        """
        randomly generate date value in the
        'dd/mm/yyyy' format
        """
        year, mon = self.random_year(), self.random_month()
        day = self.random_day(self.check_max_days(year, mon))
        return "{d}/{m}/{y}".format(d=str(day).zfill(2), m=str(mon).zfill(2), y=year)


if __name__ == "__main__":
    print(RandomDate().generate_random_date())
