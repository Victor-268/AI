import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probabilities = {}

     # Total number of pages in the corpus
    total_pages = len(corpus)

    # Number of outgoing links from the current page
    outgoing_links = len(corpus[page]) if corpus[page] else total_pages

    # Probability for choosing a link from the current page
    link_probability = damping_factor / outgoing_links

    # Probability for choosing a random page from the corpus
    random_probability = (1 - damping_factor) / total_pages

    # Iterate through all pages in the corpus
    for p in corpus:
        # If the current page has outgoing links
        if p in corpus[page]:
            probabilities[p] = link_probability + random_probability
        else:
            probabilities[p] = random_probability

    return probabilities

    #The function accepts three arguments: corpus, page, and damping_factor.
    #    The corpus is a Python dictionary mapping a page name to a set of all pages linked to by that page.
    #    The page is a string representing which page the random surfer is currently on.
    #    The damping_factor is a floating point number representing the damping factor to be used when generating the probabilities.
    #The return value of the function should be a Python dictionary with one key for each page in the corpus. Each key should be mapped to a value representing the probability that a random surfer would choose that page next. The values in this returned probability distribution should sum to 1.
    #    With probability damping_factor, the random surfer should randomly choose one of the links from page with equal probability.
    #    With probability 1 - damping_factor, the random surfer should randomly choose one of all pages in the corpus with equal probability.
    #For example, if the corpus were {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}, the page was "1.html", and the damping_factor was 0.85, then the output of transition_model should be {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}. This is because with probability 0.85, we choose randomly to go from page 1 to either page 2 or page 3 (so each of page 2 or page 3 has probability 0.425 to start), but every page gets an additional 0.05 because with probability 0.15 we choose randomly among all three of the pages.
    #If page has no outgoing links, then transition_model should return a probability distribution that chooses randomly among all pages with equal probability. (In other words, if a page has no links, we can pretend it has links to all pages in the corpus, including itself.)

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_counts = {page: 0 for page in corpus}

    # Randomly select a starting page
    current_page = random.choice(list(corpus.keys()))

    # Iterate through n samples
    for _ in range(n):
        # Increment count for current page
        page_counts[current_page] += 1

        # Generate transition model for current page
        probabilities = transition_model(corpus, current_page, damping_factor)

        # Choose next page based on transition model
        current_page = random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]

    # Normalize counts to obtain PageRank values
    total_samples = sum(page_counts.values())
    pagerank = {page: count / total_samples for page, count in page_counts.items()}

    return pagerank


    #n is an integer representing the number of samples that should be generated to estimate PageRank values.
    #The return value of the function should be a Python dictionary with one key for each page in the corpus. Each key should be mapped to a value representing that page’s estimated PageRank (i.e., the proportion of all the samples that corresponded to that page). The values in this dictionary should sum to 1.
    #The first sample should be generated by choosing from a page at random.
    #For each of the remaining samples, the next sample should be generated from the previous sample based on the previous sample’s transition model.
    #You will likely want to pass the previous sample into your transition_model function, along with the corpus and the damping_factor, to get the probabilities for the next sample.
    #For example, if the transition probabilities are {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}, then 5% of the time the next sample generated should be "1.html", 47.5% of the time the next sample generated should be "2.html", and 47.5% of the time the next sample generated should be "3.html".
    #You may assume that n will be at least 1.

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #The function accepts two arguments: corpus and damping_factor.
    #The corpus is a Python dictionary mapping a page name to a set of all pages linked to by that page.
    #The damping_factor is a floating point number representing the damping factor to be used in the PageRank formula.


    #The return value of the function should be a Python dictionary with one key for each page in the corpus. Each key should be mapped to a value representing that page’s PageRank. The values in this dictionary should sum to 1.
    #The function should begin by assigning each page a rank of 1 / N, where N is the total number of pages in the corpus.
    #The function should then repeatedly calculate new rank values based on all of the current rank values, according to the PageRank formula in the “Background” section. (i.e., calculating a page’s PageRank based on the PageRanks of all pages that link to it).
    #A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself).
    #This process should repeat until no PageRank value changes by more than 0.001 between the current rank values and the new rank values.

    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}  # Initialize PageRank values

    while True:
        new_pagerank = {}  # Store the new PageRank values
        dangling_value = 0  # Total PageRank from dangling pages

        # Calculate new PageRank values for each page
        for page in corpus:
            new_pagerank[page] = (1 - damping_factor) / N  # Start with the damping factor term

            # Iterate through pages linking to the current page
            for linking_page, linked_pages in corpus.items():
                if page in linked_pages:
                    num_links = len(linked_pages)
                    new_pagerank[page] += damping_factor * pagerank[linking_page] / num_links

            # Accumulate PageRank from dangling pages
            if len(corpus[page]) == 0:
                dangling_value += pagerank[page]

        # Distribute dangling PageRank equally among all pages
        dangling_contribution = damping_factor * dangling_value / N
        for page in corpus:
            new_pagerank[page] += dangling_contribution

        # Check for convergence
        max_diff = max(abs(new_pagerank[page] - pagerank[page]) for page in corpus)
        if max_diff < 0.001:
            break

        pagerank = new_pagerank  # Update PageRank values

    return pagerank

if __name__ == "__main__":
    main()
