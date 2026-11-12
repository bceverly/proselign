# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Writing published on the web: every post of a blog, or one article.

This is the only place training touches the network, and only when the user
names a URL.  Nothing is crawled: proselign reads the page it is given and,
for a site's front page, the post index that site itself advertises -- the
WordPress REST API, or else its RSS or Atom feed (a feed URL can also be
given directly).  Any other URL is read as a single article.

Fetched posts are kept as plain text in the voice's own private directory,
so ``train --append`` can re-measure them without going back online, and
``delete`` removes them with the voice.  Pages are untrusted data: responses
are size-capped, redirects may only lead to http(s), feeds with a DTD are
refused (no entity expansion), and ``offline`` allows loopback hosts only.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

# A feed is parsed only after any DTD has been refused, so no entity can expand.
from xml.etree import ElementTree  # nosec B405 # nosemgrep

from proselign import __version__
from proselign.errors import FetchError, UsageError
from proselign.inference.openai_compat import is_loopback
from proselign.ingest.formats.html import html_text, html_title
from proselign.paths import ensure_private_dir, write_private_file

WEB_DIR = "web"
TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
PER_PAGE = 100
MAX_API_PAGES = 100  # 10,000 posts
WP_API_REL = "https://api.w.org/"
_FEED_TYPES = ("application/rss+xml", "application/atom+xml")
_ATOM = "{http://www.w3.org/2005/Atom}"
_CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"


@dataclass(frozen=True)
class Response:
    """One fetched resource: its final URL, lower-cased headers and body."""

    url: str
    headers: dict[str, str]
    body: bytes

    def text(self) -> str:
        """The body decoded with its declared charset (UTF-8 by default)."""
        match = re.search(r"charset=([\w.-]+)", self.headers.get("content-type", ""), re.I)
        try:
            return self.body.decode(match.group(1) if match else "utf-8", errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class Post:
    """One piece of published writing."""

    url: str
    title: str
    text: str


Getter = Callable[[str, str], Response]


def is_url(source: str) -> bool:
    """True when a ``train`` source names a web page rather than a path."""
    return source.lower().startswith(("http://", "https://"))


def check_url(url: str, offline: bool) -> None:
    """Refuse anything but http(s), and non-loopback hosts when ``offline``."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise UsageError(f"not an http(s) URL: {url!r}")
    if offline and not is_loopback(url):
        raise UsageError(f"offline mode forbids fetching {url}")


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follows redirects only to http(s) URLs the fetcher would accept."""

    def __init__(self, offline: bool) -> None:
        super().__init__()
        self.offline = offline

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, self.offline)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_getter(offline: bool) -> Getter:
    """A size-capped HTTP GET that turns every failure into :class:`FetchError`."""
    opener = urllib.request.build_opener(_Redirects(offline))

    def get(url: str, accept: str) -> Response:
        headers = {"User-Agent": f"proselign/{__version__}", "Accept": accept}
        # The scheme was restricted to http(s) by check_url.
        request = urllib.request.Request(url, headers=headers)  # nosec B310
        try:
            with opener.open(request, timeout=TIMEOUT_SECONDS) as reply:
                body = reply.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise FetchError(f"{url} is larger than {MAX_RESPONSE_BYTES >> 20} MB")
                found = {k.lower(): ", ".join(reply.headers.get_all(k)) for k in reply.headers}
                return Response(reply.geturl(), found, body)
        except urllib.error.HTTPError as exc:
            raise FetchError(f"cannot fetch {url}: HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise FetchError(f"cannot fetch {url}: {exc.reason}") from exc
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise FetchError(f"cannot fetch {url}: {exc}") from exc

    return get


@dataclass
class Fetcher:
    """Reads posts from the web; ``getter`` is replaceable for tests."""

    offline: bool = False
    getter: Getter | None = None
    notes: list[str] = field(default_factory=list)

    def get(self, url: str, accept: str = "*/*") -> Response:
        """Fetch ``url`` after checking it is allowed."""
        check_url(url, self.offline)
        return (self.getter or http_getter(self.offline))(url, accept)

    def posts(self, url: str) -> list[Post]:
        """Every post of the blog at ``url``, or the one article it names."""
        page = self.get(url, "text/html, application/xhtml+xml, application/xml;q=0.9")
        if _looks_like_feed(page):
            return self._from_feed(page)
        markup = page.text()
        links = _links(markup)
        api = _wordpress_root(page, links)
        if api and _is_front_page(page.url, api):
            found = self._from_wordpress(api)
            if found:
                return found
        feed = _feed_link(links, page.url)
        if feed and _is_front_page(page.url, None):
            found = self._from_feed(self.get(feed, ", ".join(_FEED_TYPES)))
            if found:
                self.notes.append(f"{url}: read from its feed, which may hold only recent posts")
                return found
        text = html_text(markup, whole_page=True)
        return [Post(page.url, html_title(markup), text)] if text else []

    def _from_wordpress(self, api: str) -> list[Post]:
        found: list[Post] = []
        for number in range(1, MAX_API_PAGES + 1):
            params = {"per_page": PER_PAGE, "page": number, "_fields": "link,title,content"}
            reply = self.get(_api_url(api, "/wp/v2/posts", params), "application/json")
            try:
                items = json.loads(reply.text())
            except json.JSONDecodeError as exc:
                raise FetchError(f"{reply.url}: not a WordPress post list ({exc})") from exc
            if not isinstance(items, list):
                raise FetchError(f"{reply.url}: not a WordPress post list")
            found.extend(post for post in map(_wordpress_post, items) if post is not None)
            try:
                pages = int(reply.headers.get("x-wp-totalpages", number))
            except ValueError:
                pages = number
            if number >= pages or not items:
                break
        return found

    def _from_feed(self, reply: Response) -> list[Post]:
        if b"<!DOCTYPE" in reply.body or b"<!ENTITY" in reply.body:
            raise FetchError(f"{reply.url}: feeds with a DTD are refused")
        try:
            root = ElementTree.fromstring(reply.body)  # nosec B314 # DTD refused above
        except ElementTree.ParseError as exc:
            raise FetchError(f"{reply.url}: unreadable feed ({exc})") from exc
        entries = root.findall("./channel/item") + root.findall(f"{_ATOM}entry")
        found = []
        for entry in entries:
            post = _feed_post(entry, reply.url)
            if post is not None:
                found.append(post)
        return found


def _wordpress_post(item: object) -> Post | None:
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    title = item.get("title")
    link = item.get("link")
    if not isinstance(content, dict) or content.get("protected"):
        return None  # password-protected posts have no readable content
    rendered = content.get("rendered")
    if not isinstance(rendered, str) or not isinstance(link, str):
        return None
    heading = title.get("rendered", "") if isinstance(title, dict) else ""
    text = html_text(rendered)
    return (
        Post(link, html_text(heading) if isinstance(heading, str) else "", text) if text else None
    )


def _feed_post(entry: ElementTree.Element, base: str) -> Post | None:
    if entry.tag.startswith(_ATOM):
        link = next(
            (
                e.get("href", "")
                for e in entry.findall(f"{_ATOM}link")
                if e.get("rel", "alternate") == "alternate"
            ),
            "",
        )
        title = entry.findtext(f"{_ATOM}title", "")
        body = entry.findtext(f"{_ATOM}content") or entry.findtext(f"{_ATOM}summary", "")
    else:
        link = entry.findtext("link", "")
        title = entry.findtext("title", "")
        body = entry.findtext(_CONTENT) or entry.findtext("description", "")
    text = html_text(body or "")
    if not text:
        return None
    return Post(urljoin(base, link.strip()) if link else base, html_text(title or ""), text)


def _looks_like_feed(page: Response) -> bool:
    kind = page.headers.get("content-type", "").lower()
    if any(t in kind for t in _FEED_TYPES):
        return True
    head = page.body[:512].lstrip().lower()
    return head.startswith(b"<?xml") and (b"<rss" in page.body[:4096] or b"<feed" in head)


class _LinkTags(HTMLParser):
    """Collects the ``<link>`` tags of a page as attribute dictionaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "link":
            self.links.append({k: (v or "") for k, v in attrs})


def _links(markup: str) -> list[dict[str, str]]:
    parser = _LinkTags()
    parser.feed(markup)
    parser.close()
    return parser.links


def _wordpress_root(page: Response, links: list[dict[str, str]]) -> str | None:
    """The site's REST API root, from its ``Link`` header or ``<link>`` tag."""
    for match in re.finditer(r"<([^>]+)>\s*;\s*rel=\"?([^\";,]+)", page.headers.get("link", "")):
        if match.group(2) == WP_API_REL:
            return urljoin(page.url, match.group(1))
    for link in links:
        if link.get("rel") == WP_API_REL and link.get("href"):
            return urljoin(page.url, link["href"])
    return None


def _feed_link(links: list[dict[str, str]], base: str) -> str | None:
    """The site's post feed (not its comments feed)."""
    for link in links:
        href = link.get("href", "")
        if (
            "alternate" in link.get("rel", "").split()
            and link.get("type", "").lower() in _FEED_TYPES
            and href
            and "comment" not in (href + link.get("title", "")).lower()
        ):
            return urljoin(base, href)
    return None


def _is_front_page(url: str, api: str | None) -> bool:
    """Whether ``url`` is the site's home page rather than one article.

    WordPress names its home through the API root (``/blog/wp-json/`` means
    the blog lives at ``/blog/``); otherwise only the bare host counts.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if api and "rest_route=" not in api:
        home = urlparse(api).path.rstrip("/")
        home = home[: -len("/wp-json")] if home.endswith("/wp-json") else home
        return path == home and not parsed.query
    return path == "" and not parsed.query


def _api_url(api: str, route: str, params: dict) -> str:
    parsed = urlparse(api)
    if "rest_route=" in parsed.query:
        query = dict(parse_qsl(parsed.query))
        query["rest_route"] = query.get("rest_route", "").rstrip("/") + route
        return urlunparse(parsed._replace(query=urlencode({**query, **params})))
    return f"{api.rstrip('/')}{route}?{urlencode(params)}"


def save_posts(posts: list[Post], directory: Path) -> dict[Path, str]:
    """Store each post as a private text file; returns file -> post URL.

    A post's file is named after its URL, so fetching it again replaces it.
    """
    ensure_private_dir(directory)
    saved: dict[Path, str] = {}
    for post in posts:
        name = hashlib.sha256(post.url.encode("utf-8")).hexdigest()[:32] + ".txt"
        body = f"{post.title}\n\n{post.text}\n" if post.title else f"{post.text}\n"
        write_private_file(directory / name, body.encode("utf-8"))
        saved[directory / name] = post.url
    return saved
