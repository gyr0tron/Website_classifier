from unittest import TestCase
import unittest
from testing_file.website_category_classifier_testing import classifier
import json

with open("testing.json") as json_data:
    data = json.load(json_data)
class TestClassifier(TestCase):
    # global data,i,hash_values


    def test_website_classifier_news(self):
        keywords=''.join(data["News"])
        print(classifier(keywords))
        self.assertEqual(classifier(keywords),"News")

    def test_website_classifier_Science(self):
        keywords = ''.join(data["Science"])
        print(classifier(keywords))
        self.assertEqual(classifier(keywords), "Science")
    def test_website_classifier_Health(self):
        keywords = ''.join(data["Health"])
        print(classifier(keywords))
        self.assertEqual(classifier(keywords), "Health")
    def test_website_classifier_Shopping(self):
        keywords = ''.join(data["Shopping"])
        print(classifier(keywords))
        self.assertEqual(classifier(keywords), "Shopping")
    def test_website_classifier_Sports(self):
        keywords = ''.join(data["Sports"])
        print(classifier(keywords))
        self.assertEqual(classifier(keywords), "Sports")

if __name__ == '__main__':
            unittest.main()