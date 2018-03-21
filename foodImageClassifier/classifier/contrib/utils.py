# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:03:06 2018

@author: Rakshit
"""

import urllib
import urllib.request
from bs4 import BeautifulSoup


def ingredient(dish):
    ingredients=[];
    dish=dish.lower()
    if(dish=="samosa"):
        url="http://foodviva.com/snacks-recipes/samosa-recipe/"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('table',{"class":"css_fv_recipe_table"}):
            for data in link.findAll('tr'):
                #print(data.text)
                ingredients.append(data.text)

    elif(dish=="gulab jamun"):
        url="http://foodviva.com/desserts-sweets-recipes/gulab-jamun-recipe/"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('table',{"class":"css_fv_recipe_table"}):
            for data in link.findAll('tr'):
                #print(data.text)
                ingredients.append(data.text)

    elif(dish=="biryani"):
        url="https://recipes.timesofindia.com/recipes/chicken-biryani/rs53096628.cms"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('li',{"class":"clearfix"}):
            #print(link.text)
            ingredients.append(link.text)

    elif(dish=="tandoori chicken"):
        url="https://www.sanjeevkapoor.com/Recipe/Tandoori-Chicken-Cooking-with-Olive-Oil.html"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('ul',{"class":"list-unstyled"}):
            for data in link.findAll('li'):
                #print(data.text)
                ingredients.append(data.text)

    elif(dish=="momo (food)"):
        url="https://food.ndtv.com/recipe-vegetable-momos-99111"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('div',{"class":"ingredients"}):
            for data in link.findAll('li'):
                #print(data.text)
                ingredients.append(data.text)

    elif(dish=="dhokla"):
        url="http://foodviva.com/snacks-recipes/khaman-dhokla-recipe/"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('table',{"class":"css_fv_recipe_table"}):
            for data in link.findAll('tr'):
                #print(data.text)
                ingredients.append(data.text)

    elif(dish=="jalebi"):
        url="http://foodviva.com/desserts-sweets-recipes/jalebi/"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('table',{"class":"css_fv_recipe_table"}):
            for data in link.findAll('tr'):
                #print(data.text)
                ingredients.append(data.text)

    elif(dish=="pani puri"):
        url="http://foodviva.com/snacks-recipes/pani-puri/"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('table',{"class":"css_fv_recipe_table"}):
            for data in link.findAll('tr'):
                #print(data.text)
                ingredients.append(data.text)

    elif(dish=="dosa"):
        url="http://foodviva.com/south-indian-recipes/masala-dosa/"
        page= urllib.request.urlopen(url)
        soup= BeautifulSoup(page,"html.parser")
        for link in soup.findAll('table',{"class":"css_fv_recipe_table"}):
            for data in link.findAll('tr'):
                #print(data.text)
                ingredients.append(data.text)

    return ingredients[1:]

if __name__ == "__main__":
    dishes = ['samosa', 'gulab jamun', 'biryani', 'tandoori chicken', 'momo (food)',
              'dhokla', 'jalebi', 'pani puri', 'dosa']

    for dish in dishes:
        print("Testing -> ", dish)
        assert 0 < len(ingredient(dish))

    print("----------------- Test passed ---------------------- Ok")
