# MY BONUS QUESTION

# Grabbing only Bob Marley's quotes
bob_marley = []
albert_ein = []
n = 1

while True:
#     print(f"\n{n}")
    site = requests.get(f"http://quotes.toscrape.com/page/{n}/")
    soup = bs4.BeautifulSoup(site.text, 'lxml')

    # quote section
    quote_tags = soup.select(".quote")
    for auth_tag in quote_tags:
    #     print(auth_tag)
        author_tags = auth_tag.select('.author')

        for auth in author_tags:
            if auth.text.lower() == 'bob marley':
                bob_marley.append(auth_tag.text)

            elif auth.text.lower() == 'albert einstein':
                albert_ein.append(auth_tag.text)

    next_tags = soup.select(".next")
    if len([t.text for t in next_tags]) < 1:
        print(f"Last Page: {n}")
        break

    n += 1

search = bob_marley
print(f"\nThere are {len(search)} Bob Marley quotes")
[print(f"\n{q}") for q in search]

search = albert_ein
print(f"\nThere are {len(search)} Albert Einstein quotes")
[print(f"\n{q}") for q in albert_ein]
