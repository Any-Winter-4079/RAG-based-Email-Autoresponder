##############################################
# Helper 1: Fetch content from URL with Jina #
##############################################
def fetch_with_jina(url, timeout):
    import requests
    from urllib.parse import urldefrag

    url_without_fragment = urldefrag(url)[0]
    jina_url = f"https://r.jina.ai/{url_without_fragment}"
    
    try:
        response = requests.get(jina_url, timeout=timeout)
        if response.status_code != 200:
            print(f"fetch_with_jina: failed to fetch {url}: status code {response.status_code}")
            return None
        
        content = response.content.decode("utf-8", errors="replace")
        return content.strip()

    except Exception as e:
        print(f"fetch_with_jina: error fetching {url}: {str(e)}")
        return None

####################################
# Helper 2: Clean up Jina Markdown #
####################################
def clean_up_jina_markdown(content):
    import re

    # remove "Markdown Content" label
    content = re.sub(r"^Markdown Content:.*?\n", "", content, flags=re.MULTILINE | re.IGNORECASE)

    # remove menu(s)
    content = re.sub(r"^.*Alternar menú.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^Main Menu\s*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^MENÚ DE NAVEGACIÓN\s*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^Navigation Menu\s*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    
    # remove "skip to content" style links
    content = re.sub(r"\[.*?content.*?\]\(.*?\)", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\[.*?contenido.*?\]\(.*?\)", "", content, flags=re.IGNORECASE)

    # collapse 3 or more (often, a lot of) dashes into 5 dashes
    content = re.sub(r"^\s*-{3,}\s*$", "-----", content, flags=re.MULTILINE)

    # collapse 3 or more (often, a lot of) equals into 5 equals
    content = re.sub(r"^\s*={3,}\s*$", "=====", content, flags=re.MULTILINE)
    
    # remove images
    content = re.sub(r"\[\!\[.*?\]\(.*?\)\]\(.*?\)|\!\[.*?\]\(.*?\)", "", content)

    # remove footer "junk"
    content = re.sub(r"^.*Copyright ©.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^.*Powered by.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^.*Sitio creado por.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^.*Aviso legal.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^.*Legal Notice.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^.*Scroll al inicio.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)

    # trim whitespace from every line
    content = "\n".join([line.strip() for line in content.split('\n')])

    # remove lines that are only a bullet point + a link:
    # pattern start -> whitespace -> * -> whitespace -> [Text](Url) -> whitespace -> end
    # (preserves lists that have text after the link, but kills navigation lists)
    content = re.sub(r"^\s*\*\s*\[[^\]]+\]\([^\)]+\)\s*$", "", content, flags=re.MULTILINE)

    # remove empty links []()
    content = re.sub(r"\[\]\(.*?\)", "", content)

    # remove lines that are only an exclamation mark
    content = re.sub(r"^\s*!\s*$", "", content, flags=re.MULTILINE)

    # remove lines that are only a bullet point (with optional whitespace)
    content = re.sub(r"^\s*\*\s*$", "", content, flags=re.MULTILINE)

    # remove pagination 
    # matches: << 1 2 3 >> or < 1 2 3 >
    content = re.sub(r"^\s*<<[\s\d]+>>\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*<[\s\d]+>\s*$", "", content, flags=re.MULTILINE)

    # remove lines that became empty after trimming
    content = re.sub(r"\n{3,}", "\n\n", content) # max 2 newlines
    
    # remove lines that are ONLY exclamation marks
    content = re.sub(r"^\s*!+\s*$", "", content, flags=re.MULTILINE)

    return content.strip()

###############################################################
# Helper 3: Check if URL has text or is image, excluded, etc. #
###############################################################
def is_content_url(url, excluded_urls, gsfs_base_url, allowed_gsfs_urls):
    from urllib.parse import urlparse

    parsed_url = urlparse(url)
    if url in excluded_urls:
        return False
    if "upm.es" not in parsed_url.netloc:
        return False
    if "/en/" in url:
        return False
    if url.startswith(gsfs_base_url) and url not in allowed_gsfs_urls:
        return False
    if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico")):
        return False
    
    return True

#############################################
# Helper 4: Extract URLs from (URL) content #
#############################################
def extract_urls(content, excluded_urls, gsfs_base_url, allowed_gsfs_urls):
    import re
    from urllib.parse import urldefrag

    if not content:
        return set()
    
    url_pattern = re.compile(r'https?://[^\s()<>"\'\[\]]+')
    urls = set(url_pattern.findall(content))
    return {urldefrag(u)[0] for u in urls if is_content_url(urldefrag(u)[0], 
                                                            excluded_urls,
                                                            gsfs_base_url,
                                                            allowed_gsfs_urls)}

#########################################
# Helper 5: Remove '/es/' from URL path #
#########################################
def remove_es_from_path(url):
    from urllib.parse import urlparse, urlunparse

    # remove leading /es from the URL path, keeps everything else (domain, query, etc.)
    parsed = urlparse(url)
    path = parsed.path
    if path.startswith("/es/"):
        path = path[3:]
    normalized = parsed._replace(path=path)
    return urlunparse(normalized)

#############################################
# Helper 6: Extract URLs from (URL) content #
#############################################
def crawl(
        start_url,
        additional_urls,
        max_depth,
        max_links_per_page,
        timeout,
        excluded_urls,
        gsfs_base_url,
        allowed_gsfs_urls
        ):
    import time
    import random

    url_content_dict = {}
    visited_urls = set()
    normalized_visited_urls = set()
    to_visit_urls = {start_url}

    # visit source_url and inside links up to max depth
    for depth in range(max_depth):
        print(f"crawl: depth {depth + 1} -> {len(to_visit_urls)} URLs to visit")
        next_level_urls = set()

        for url in to_visit_urls:
            # skip URL if visited
            if url in visited_urls:
                continue

            normalized_url = remove_es_from_path(url)
            # skip URL if visited in normalized form
            if normalized_url in normalized_visited_urls:
                print(f"crawl: skipping URL due to being a normalized duplicate: {url}")
                continue

            # add URL to visited
            visited_urls.add(url)
            normalized_visited_urls.add(normalized_url)

            # fetch URL content
            print(f"crawl: fetching from: {url}")
            content = fetch_with_jina(url, timeout)

            if content:
                manually_cleaned_content = clean_up_jina_markdown(content)

                # add URL content
                url_content_dict[url] = {
                    "raw": {
                        "text": content
                    },
                    "manually_cleaned": {
                        "text": manually_cleaned_content
                    }
                }

                # extract URLs from content
                new_urls = extract_urls(content, excluded_urls, gsfs_base_url, allowed_gsfs_urls)

                # add new URLS (up to max depth)
                links_added = 0
                for new_url in sorted(new_urls):
                    if links_added >= max_links_per_page:
                        break
                    if new_url not in visited_urls and new_url not in next_level_urls:
                        next_level_urls.add(new_url)
                        links_added += 1

            # give time between requests
            time.sleep(random.uniform(1, 10))

        to_visit_urls = next_level_urls

    # visit additional urls
    if additional_urls:
        print(f"crawl: processing {len(additional_urls)} additional URLs (depth 1)")
        for url in additional_urls:
            # skip URL if visited
            if url in visited_urls:
                continue
            
            normalized_url = remove_es_from_path(url)
            # skip URL if visited in normalized form
            if normalized_url in normalized_visited_urls:
                print(f"crawl: skipping additional URL due to being a normalized duplicate: {url}")
                continue

            # add URL to visited
            visited_urls.add(url)
            normalized_visited_urls.add(normalized_url)

            # fetch additional URL content
            print(f"crawl: fetching additional URL: {url}")
            content = fetch_with_jina(url, timeout)

            if content:
                manually_cleaned_content = clean_up_jina_markdown(content)

                # add additional URL content
                url_content_dict[url] = {
                    "raw": {
                        "text": content
                    },
                    "manually_cleaned": {
                        "text": manually_cleaned_content
                    }
                }
            
            # give time between requests
            time.sleep(random.uniform(1, 10))

    print(f"crawl: crawled {len(url_content_dict)} pages")
    return url_content_dict