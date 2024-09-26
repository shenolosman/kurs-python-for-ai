# bookCopy1 = {
#     "title": "The Fellopship of the ring",
#     "authoer": "J.R.R Tolkien",
#     "year": 1954,
#     "genre": "fantasy",
#     "borrowed": 1,
# }
# bookCopy2 = {
#     "title": "Kill a mockiungbird",
#     "authoer": "harper lee",
#     "year": 1960,
#     "genre": "fiction",
#     "borrowed": 0,
# }

# library = [bookCopy1, bookCopy2]

# for book in library:
#     print(book["title"])

from functools import reduce
import itertools


class Book:
    def __init__(self, title, author, year, genre, borrowed, pages):
        self.title = title
        self.author = author
        self.year = year
        self.genre = genre
        self.is_borrowed = borrowed
        self.pages = pages

    def __str__(self):
        return f"{self.title} by {self.author} ({self.year})"

    def borrow(self):
        if self.is_borrowed:
            return False
        self.is_borrowed = True
        return True

    def return_book(self):
        self.is_borrowed = False


class Library:
    def __init__(self, books=None):
        if books is None:
            books = []
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def remove_book(self, book):
        self.books.remove(book)

    def find_book(self, book):
        return next(
            (book for book in self.books if book.title.lower() == title.lower()), None
        )

    def list_books(self):
        return list(map(str, self.books))

    def available_books(self):
        return list(filter(lambda book: not book.is_borrowed, self.books))

    def get_total_pages(self):
        return reduce(lambda x, y: x + y, map(lambda book: book.pages, self.books), 0)

    def group_by_genre(self):
        return {
            genre: list(books)
            for genre, books in itertools.groupby(
                sorted(self.books, key=lambda x: x.genre), key=lambda x: x.genre
            )
        }


bookObj = Book("kucuk prens", "senol", 1999, "klasik", False, 232)
print(bookObj)
print(bookObj.borrow())
print(bookObj.is_borrowed)

bookObj.return_book()
print(bookObj.is_borrowed)


library = Library()
library.add_book(bookObj)
library.add_book(Book("1984", "George Orwell", 1945, "fact", False, 312))
library.add_book(Book("Kill a mockingbird", "harper lee", 1925, "fictions", False, 432))

print(library.books)

print("--- list --------")
print(library.list_books())

print("--- list --------")
list(map(print, library.list_books()))


if book_to_borrow := library.find_book("1984"):
    if book_to_borrow.borrow():
        print(f"\nBorrowed: {book_to_borrow}")
    else:
        print(f"\n{book_to_borrow} is already borrowed.")
