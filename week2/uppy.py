# Exempeluppgift 1 - class Person
# Skapa en enkel klass Person med följande steg:
# Definiera klassen Person. Lägg till en konstruktor (__init__-metod) som tar emot name och age som parametrar. Skapa en metod introduce() som returnerar en presentation av personen. Skapa några instanser av Person och anropa introduce() metoden för var och en.

class Person:
    def __init__(self,name,age):
            self.name=name
            self.age=age
            self.hobbies=[]
    
    def introduce(self):
        return f"Hi, my name is {self.name} and i am {self.age} years old."
            




# Exempeluppgift del 2 - Attribut och metoder för Person
# Utöka Person-klassen från föregående uppgift:

# Lägg till ett attribut hobbies som en lista i konstruktorn.
# Skapa en metod add_hobby(hobby) för att lägga till en hobby.
# Skapa en metod get_hobbies() som returnerar en sträng med alla hobbies.
# Skriv över/overwritea __str__ metoden för att ge en fin strängrepresentation av objektet, när man till exempel printar ett Person-objekt.

    def add_hobby(self,hobby):
        self.hobbies.append(hobby)

    def get_hobbies(self):
        if self.hobbies:
            return f"Mina hobbies är : {",".join(self.hobbies)}."
        else:
            return "Jag har inga hobbies än."
    
    def __str__(self):
        return f"{self.name}, {self.age} år gammal."

person1= Person("Shenol",36)
person1.add_hobby("Måla")
print(person1.introduce())
print(person1.get_hobbies())

print(person1)







#---------- 