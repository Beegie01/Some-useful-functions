class Vehicle:
    def __init__(self, name, seats, wheels, speed):
        self.name = name
        self.speed = speed
        self.seats = seats
        self.wheels = wheels
        self.color = None

    def set_color(self, color):
        self.color = color.capitalize() if type(color) == str else None

    def get_vehicle(self):
        print(''.join([f"{self.color}-colored {self.name} is a {self.seats}-seated {self.wheels}-wheel drive with maxspeed of {self.speed}"
               if self.color is not None else f"{self.name} is a {self.seats}-seated {self.wheels}-wheel drive with maxspeed of {self.speed}"]))

car = Vehicle('Car', 6, 'four', '200mph')
car.get_vehicle()
car.set_color(34)
car.get_vehicle()


class Person:
    def __init__(self, name, age, race):
        self.name = name
        self.race = race
        self.age = age

    def grooming(self):
        print('Now grooming')

    # the '@property' decorator helps to transform
    # a method into an attribute-like entity
    # and it cannot be changed
    @property  # this method works like an attribute and it cannot be changed
    def inhale(self):
        return 'breathe in'

    @property
    def exhale(self):
        return 'breathe out'

class Male(Person):
    def __init__(self, name, race, age, marital_status):
        # INHERITANCE: Male class inherits attr and methods from Person class
        Person.__init__(self, name, age, race)

        # ABSTRACTION AND ENCAPSULATION
        self.father = None # public attribute
        self.status = marital_status # public attribute
        self._worth = None # protected attr: hidden, but still accessible outside of the class (user has indirect access)
        self.__hiv = None  # private attr: not accessible outside the class (user has no access)

    # POLYMORPHISM: shares the same grooming method as parent class
    # but both have different behaviours
    def grooming(self):
        print(f"{self.name} is now at the barbershop")

    def set_hidden_attr(self, hiv_status: str, net_worth: float):
        self.__hiv = hiv_status.capitalize() if type(hiv_status) == str else None
        self._worth = net_worth if type(net_worth) in [float, int] else None
        print(f"{''.join([f'{k} has been set to {v}' for k, v in self.__dict__.items() if k == '_worth'])}\n")
        print(f"{''.join([f'{k} has been set to {v}' for k, v in self.__dict__.items() if k == '__hiv'])}\n")

boy = Male(marital_status='single', name='Clint', age=14, race='black')

print(boy.__dict__)

boy.grooming()
boy.set_hidden_attr('negative', 12000)
# boy.
print(boy.__dict__)
print("Boy's respiration:\n{} and {}".format(boy.inhale, boy.exhale))

man = Male(marital_status='married', name='Dr. Osagie', age=34, race='black')
print(man.__dict__)

man.set_hidden_attr(23, 'one billion pounds')
print(man.inhale)