import arxiv

def get_arxiv_papers(topic, max_results=5):

    client = arxiv.Client()

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in client.results(search):

        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": str(result.published.date()),
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })

    return papers
