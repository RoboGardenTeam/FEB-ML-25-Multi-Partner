class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
    def display_info(self):
        print(f"Make: {self.make}")
        print(f"Model: {self.model}")
        print(f"Year: {self.year}")
class BMW(Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.make = 'BMW'
    def display_info(self):
        print(f"Make: {self.make}")
        print(f"Model: {self.model}")
        print(f"Year: {self.year}")
    
car1 = Car('Honda', 'Civic', 2022)
car2 = BMW('Honda', 'Accord', 2022)
print(car1.display_info())
print(car2.display_info())