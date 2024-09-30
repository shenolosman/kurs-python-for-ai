import random

#upp1 
# write_in=input("Skriva in text :")
# print(len(write_in))


#upp2
# write_in=input("Give me a word : ")
# myObj={}
# for char in write_in:
#     if(char not in myObj.keys()):
#         myObj[char]=1
#     else:
#         myObj[char]+=1
    
# print(myObj)
# print("------------")
# for key, value in myObj.items():
#     print(f"{key}: #{value}")

#upp3
# write_in=input("give me a word : ")

# first_two_letter=write_in[0:2]
# last_two_letter=write_in[-2:]

# print(first_two_letter,last_two_letter)

#upp4
# first_string = list(input("Enter the first string: "))
# second_string = list(input("Enter the second string: "))

# temp_character = first_string[0]    # Sparar ner första strängens första tecken i en "temporär" variabel
# first_string[0] = second_string[0]  # Byter värdet på första strängens första tecken
# second_string[0] = temp_character   # Byter värdet på andra strängens första tecken

# temp_character = first_string[1]    # Upprepar processen för strängarnas andra (second) tecken
# first_string[1] = second_string[1]
# second_string[1] = temp_character

# # Vi gör listorna till strängar igen med jon(), och lägger ihop till en enda sträng som sparas ner i variabeln 'combined_string'
# combined_string = "".join(first_string) + " " + "".join(second_string)  
# print(combined_string)

#upp5
# write_in=input("give me a word : ")
# if(len(write_in)>3):
#     print(write_in+"ing")
# else:
#     print(f"{write_in} is too short to add 'ing'")

#upp6
# write_in=input("give me a word : ")
# trimWhitespace=write_in.replace(" ","").replace("\\n","").replace("\\t","")

# print("with white space :",trimWhitespace)
# trimEverything=[]

# for index in range(len(trimWhitespace)):
#     if(index%2==0):
#         trimEverything.append(trimWhitespace[index])
# print("with index trim :","".join(trimEverything))


#uppgift7
# write_in=input("give me a word : ")
# myArray=write_in.replace(" ","").split(",")
# orderedList=[]
# for x in myArray:
#     if(x not in orderedList):
#         orderedList.append(x)   
    
# orderedList.sort()
# print(", ".join(orderedList))

#upp8
# write_in=input("give me a word : ")
# first_four_chars=write_in[0:4]
# count=0

# for char in first_four_chars:
#     if char.isupper():
#         count+=1
#     else:
#         continue
#     if count>=2:
#         write_in=write_in.upper()
#         break
    
# print(write_in)

#upp9
# write_in=input("give me a word : ")
# reversed= ""
# if len(write_in)>4:
#     reversed=write_in[::-1][:-3]
# else:
#     print("shorter than 4 syllables")
# print(reversed*4)

#upp10
# def repeat_last_two(original_string):
#     if len(original_string) >= 2:
#         last_two_chars = original_string[-2:]
#         repeating_string = last_two_chars*4
#         return repeating_string
#     else:
#         return original_string

# write_in=input("give me a word : ")
# print(repeat_last_two(write_in))


#upp11
# wordList=[]
# while True:
#     new_word= input("Give me a word or stop with spacing! :")
#     if new_word=="":
#         break
#     else:
#         wordList.append(new_word)

# def longest_word(word):
#     if(word):
#         longest=max(word, key=len)
#         length=len(longest)
#         return longest,length
#     return "",0 #returns default values
    
# word,length=longest_word(wordList)
# print(f"longest word is {word} and its length is {length}")

#upp12
# def generate_multiplication_table():
#     print("    |", end="")
#     for i in range(1, 11):
#         print(f"{i:4}", end="")
#     print("\n" + "-" * 45)

#     for i in range(1, 11):
#         print(f"{i:2} |", end="")
#         for j in range(1, 11):
#             print(f"{i*j:4}", end="")
#         print()

# generate_multiplication_table()

#upp13
# write_in=int(input("Factorial number : "))

# def factorial(number):
#     if(number <= 1):
#         return 1
#     else:
#         return number*factorial(number-1)

# result=factorial(write_in)
# print(f"The factorial of number {write_in} is {result}")


#upp14
# print("---Guessing Game---")
# randomNumber=random.randrange(1,100)
# while True:
#     guessRightInput=int(input("\nGive a shot bet10ween 1-100 to guess right number :"))
#     if guessRightInput==randomNumber:
#         print(f"You guessed right. The number is {randomNumber}!")
#         break
#     elif(guessRightInput>randomNumber):
#         print("You are guessed a bit higher, try again!\n")
#         continue
#     elif(guessRightInput<randomNumber):
#         print("You are guessed a bit lower, try again!\n")
#         continue
#     else:
#         print("There is an error!")
#         break


#upp15
# def palindrome(number):
#     if (number == number[::-1]):
#         return print("Number is palindrome")
#     else:
#         return print("Number is NOT palindrome!")
        
# write_in=input("Write to test if number is palindrome :")
# print(palindrome(write_in))