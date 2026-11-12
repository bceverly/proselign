# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Training from the web: HTML prose extraction, post discovery, caching."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from proselign.errors import EXIT_FETCH, EXIT_OK, EXIT_USAGE, FetchError, UsageError
from proselign.ingest import web
from proselign.ingest.formats.html import html_text, html_title, is_code_like
from proselign.ingest.web import Fetcher, Post, Response, check_url, is_url, save_posts

from .conftest import SAMPLE_PARAGRAPHS, run

PROSE = " ".join(f"<p>{p}</p>" for p in SAMPLE_PARAGRAPHS)


# -- HTML prose ---------------------------------------------------------------


def test_html_text_keeps_prose_and_drops_code_and_furniture():
    markup = (
        "<h2>A Heading</h2><p>I wrote this &amp; meant it.</p>"
        "<pre><code>$ make install</code></pre>"
        "<blockquote><p>Someone else said this.</p></blockquote>"
        "<script>alert(1)</script><table><tr><td>cell</td></tr></table>"
        "<div class='comments'><p>A reader's comment.</p></div>"
        "<ul><li>First point</li><li>Second point</li></ul>"
        "<p>Line one of a sentence<br>that continues here.</p>"
    )
    text = html_text(markup)
    assert text.split("\n\n") == [
        "A Heading",
        "I wrote this & meant it.",
        "First point",
        "Second point",
        "Line one of a sentence that continues here.",
    ]


def test_html_text_drops_command_lines_inside_paragraphs():
    markup = (
        "<p>$ sudo apt install qemu<br>$ cd hyperion<br>export ENV=$HOME/.kshrc</p>"
        "<p>Then run “$ doas pfctl -f /etc/pf.conf” and the machine relays traffic.</p>"
    )
    assert html_text(markup) == (
        "Then run “$ doas pfctl -f /etc/pf.conf” and the machine relays traffic."
    )


@pytest.mark.parametrize(
    "line",
    [
        "$ sudo apt install qemu",
        "PS > ./dnscrypt-proxy -service start",
        "boot> boot -s",
        "fdisk: 1> quit",
        "alias ls=”colorls -G”",
        "COMMENT= library to en/decode DNS",
        "pass in on $int_if inet",
        "menuentry “Arch” {",
        "echo 'Uptime:' >> /tmp/report",
        "sudo systemctl restart rsyslog",
        "procedure, public :: get_value => value",
        "https://example.com/only-a-link",
        "/dev/ada0p5.eli none swap sw 0 0",
    ],
)
def test_code_like_lines(line):
    assert is_code_like(line)


@pytest.mark.parametrize(
    "line",
    [
        "I like short sentences.",
        "Conclusion",
        "Install Hercules v4 (the one that installs with apt is v3)",
        "Next, add the line “dns=127.0.0.1” to the main section of the file.",
        "The files land in $WRKDIR under the ports tree, which is handy.",
    ],
)
def test_prose_lines(line):
    assert not is_code_like(line)


def test_whole_page_reads_only_the_article():
    page = (
        "<html><head><title>My Post &mdash; Blog</title></head><body>"
        "<nav><p>Home About</p></nav><div class='sidebar'><p>Tag cloud.</p></div>"
        "<article><header><h1>My Post</h1></header><p>The real text.</p>"
        "<footer>Posted in Stuff</footer></article><article><p>Next post.</p></article>"
        "</body></html>"
    )
    assert html_text(page, whole_page=True) == "The real text."
    assert html_title(page) == "My Post — Blog"
    assert html_title("<p>no title</p>") == ""


def test_whole_page_falls_back_to_main_then_body():
    assert html_text("<div>x</div><main><p>Main text.</p></main>", whole_page=True) == (
        "Main text."
    )
    assert html_text("<body><p>Body text.</p></body>", whole_page=True) == "Body text."


def test_unclosed_skipped_elements_unwind():
    assert html_text("<figure><p>caption<img src=x></figure><p>After.</p>") == "After."


# -- URLs -----------------------------------------------------------------------


def test_is_url_and_check_url():
    assert is_url("https://example.com") and is_url("HTTP://example.com")
    assert not is_url("/home/me/blog") and not is_url("file:///etc/passwd")
    check_url("https://example.com/", offline=False)
    check_url("http://127.0.0.1:8000/", offline=True)
    with pytest.raises(UsageError):
        check_url("ftp://example.com/", offline=False)
    with pytest.raises(UsageError):
        check_url("https:///nohost", offline=False)
    with pytest.raises(UsageError, match="offline"):
        check_url("https://203.0.113.9/", offline=True)


# -- discovery --------------------------------------------------------------------


class FakeWeb:
    """A getter serving canned responses and recording what was asked for."""

    def __init__(self, pages: dict[str, Response]):
        self.pages = pages
        self.asked: list[str] = []

    def __call__(self, url: str, accept: str) -> Response:
        self.asked.append(url)
        if url not in self.pages:
            raise FetchError(f"cannot fetch {url}: HTTP 404 Not Found")
        return self.pages[url]


def page(url, body, **headers):
    data = body.encode("utf-8") if isinstance(body, str) else body
    return Response(url, {k.replace("_", "-"): v for k, v in headers.items()}, data)


def wp_posts(start, count):
    return json.dumps(
        [
            {
                "link": f"https://blog.example/post-{i}/",
                "title": {"rendered": f"Post {i} &amp; more"},
                "content": {"rendered": PROSE, "protected": False},
            }
            for i in range(start, start + count)
        ]
    )


API = "https://blog.example/wp-json/"
HOME = "https://blog.example/"


def wp_url(page_number):
    return f"{API}wp/v2/posts?per_page=100&page={page_number}&_fields=link%2Ctitle%2Ccontent"


def test_wordpress_front_page_reads_every_post_across_pages():
    fake = FakeWeb(
        {
            HOME: page(HOME, "<html></html>", link=f'<{API}>; rel="https://api.w.org/"'),
            wp_url(1): page(wp_url(1), wp_posts(0, 2), x_wp_totalpages="2"),
            wp_url(2): page(
                wp_url(2),
                json.dumps(
                    [
                        {"link": "x", "content": {"rendered": "<p>hidden</p>", "protected": True}},
                        "junk",
                        {"link": "y", "content": {"rendered": 3}},
                    ]
                    + json.loads(wp_posts(2, 1))
                ),
                x_wp_totalpages="2",
            ),
        }
    )
    posts = Fetcher(getter=fake).posts(HOME)
    assert [p.url for p in posts] == [f"https://blog.example/post-{i}/" for i in range(3)]
    assert posts[0].title == "Post 0 & more"
    assert posts[0].text.startswith("I don't like long meetings.")


def test_wordpress_api_found_from_link_tag_and_rest_route():
    api = "https://blog.example/?rest_route=/"
    listing = (
        "https://blog.example/?rest_route=%2Fwp%2Fv2%2Fposts&per_page=100&page=1"
        "&_fields=link%2Ctitle%2Ccontent"
    )
    fake = FakeWeb(
        {
            HOME: page(HOME, f'<link rel="https://api.w.org/" href="{api}">'),
            listing: page(listing, wp_posts(0, 1)),
        }
    )
    assert len(Fetcher(getter=fake).posts(HOME)) == 1


def test_wordpress_in_a_subdirectory():
    home = "https://example.com/blog/"
    api = "https://example.com/blog/wp-json/"
    listing = f"{api}wp/v2/posts?per_page=100&page=1&_fields=link%2Ctitle%2Ccontent"
    fake = FakeWeb(
        {
            home: page(home, "", link=f'<{api}>; rel="https://api.w.org/"'),
            listing: page(listing, wp_posts(0, 1), x_wp_totalpages="nonsense"),
        }
    )
    assert len(Fetcher(getter=fake).posts(home)) == 1


def test_wordpress_article_url_reads_only_that_article():
    url = "https://blog.example/one-post/"
    fake = FakeWeb(
        {
            url: page(
                url,
                f"<title>One</title><article>{PROSE}</article>",
                link=f'<{API}>; rel="https://api.w.org/"',
            )
        }
    )
    posts = Fetcher(getter=fake).posts(url)
    assert len(posts) == 1 and posts[0].title == "One"
    assert fake.asked == [url]


def test_bad_wordpress_listing_is_a_fetch_error():
    for body in ("not json", '{"code": "rest_disabled"}'):
        fake = FakeWeb(
            {
                HOME: page(HOME, "", link=f'<{API}>; rel="https://api.w.org/"'),
                wp_url(1): page(wp_url(1), body),
            }
        )
        with pytest.raises(FetchError, match="WordPress"):
            Fetcher(getter=fake).posts(HOME)


RSS = """<?xml version="1.0"?>
<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><title>First</title><link>/first/</link>
<content:encoded><![CDATA[{prose}]]></content:encoded></item>
<item><title>Second</title><description><![CDATA[<p>Only a summary here.</p>]]></description></item>
<item><title>Empty</title></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Atom post</title><link rel="alternate" href="https://site.example/a/"/>
<content type="html">{prose}</content></entry>
<entry><title>Summary</title><summary>Just words.</summary></entry>
</feed>"""


def test_front_page_without_wordpress_uses_its_feed():
    home = "https://site.example/"
    feed = "https://site.example/feed.xml"
    fake = FakeWeb(
        {
            home: page(
                home,
                '<link rel="alternate" type="application/rss+xml" href="/comments/feed">'
                '<link rel="alternate" type="application/rss+xml" href="/feed.xml">',
            ),
            feed: page(feed, RSS.replace("{prose}", PROSE)),
        }
    )
    fetcher = Fetcher(getter=fake)
    posts = fetcher.posts(home)
    assert [p.title for p in posts] == ["First", "Second"]
    assert posts[0].url == "https://site.example/first/"
    assert posts[1].url == feed
    assert "recent posts" in fetcher.notes[0]


def test_feed_url_given_directly():
    url = "https://site.example/atom"
    body = ATOM.replace("{prose}", "&lt;p&gt;Escaped HTML prose.&lt;/p&gt;")
    fake = FakeWeb({url: page(url, body, content_type="application/atom+xml")})
    posts = Fetcher(getter=fake).posts(url)
    assert [(p.title, p.text, p.url) for p in posts] == [
        ("Atom post", "Escaped HTML prose.", "https://site.example/a/"),
        ("Summary", "Just words.", url),
    ]


def test_feed_sniffed_without_content_type():
    url = "https://site.example/rss"
    fake = FakeWeb({url: page(url, RSS.replace("{prose}", PROSE))})
    assert len(Fetcher(getter=fake).posts(url)) == 2


def test_hostile_or_broken_feeds_are_refused():
    url = "https://site.example/rss"
    bomb = '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><rss>&a;</rss>'
    for body, reason in ((bomb, "DTD"), ("<?xml version='1.0'?><rss><channel>", "unreadable")):
        fake = FakeWeb({url: page(url, body, content_type="application/rss+xml")})
        with pytest.raises(FetchError, match=reason):
            Fetcher(getter=fake).posts(url)


def test_plain_site_front_page_is_read_as_a_page():
    home = "https://plain.example"
    fake = FakeWeb({home: page(home, f"<main>{PROSE}</main>", content_type="text/html")})
    posts = Fetcher(getter=fake).posts(home)
    assert len(posts) == 1 and posts[0].url == home
    empty = FakeWeb({home: page(home, "<script>x</script>")})
    assert not Fetcher(getter=empty).posts(home)


def test_response_text_honors_charset():
    assert page("u", "caf\xe9".encode("latin-1"), content_type="text/html; charset=latin-1").text()
    assert page("u", b"ok", content_type="text/html; charset=bogus").text() == "ok"


# -- HTTP -----------------------------------------------------------------------


class FakeReply(io.BytesIO):
    def __init__(self, body, url="https://x.example/", headers=None):
        super().__init__(body)
        self.url = url
        from email.message import Message

        self.headers = Message()
        for key, value in (headers or {"Content-Type": "text/html"}).items():
            self.headers[key] = value

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_http_getter_sends_identity_and_caps_size(monkeypatch):
    opener = FakeOpener(FakeReply(b"<p>hi</p>", headers={"Link": "<a>", "Content-Type": "t"}))
    monkeypatch.setattr(web.urllib.request, "build_opener", lambda *h: opener)
    reply = web.http_getter(False)("https://x.example/", "text/html")
    assert reply.body == b"<p>hi</p>" and reply.headers["link"] == "<a>"
    request, timeout = opener.requests[0]
    assert request.get_header("User-agent").startswith("proselign/")
    assert timeout == web.TIMEOUT_SECONDS

    monkeypatch.setattr(web, "MAX_RESPONSE_BYTES", 4)
    opener.outcome = FakeReply(b"<p>hi</p>")
    with pytest.raises(FetchError, match="larger than"):
        web.http_getter(False)("https://x.example/", "text/html")


@pytest.mark.parametrize(
    "error, message",
    [
        (urllib.error.HTTPError("u", 404, "Not Found", None, None), "HTTP 404"),
        (urllib.error.URLError("no route"), "no route"),
        (TimeoutError("timed out"), "timed out"),
    ],
)
def test_http_getter_failures_are_fetch_errors(monkeypatch, error, message):
    monkeypatch.setattr(web.urllib.request, "build_opener", lambda *h: FakeOpener(error))
    with pytest.raises(FetchError, match=message):
        web.http_getter(False)("https://x.example/", "text/html")


def test_redirects_only_to_allowed_urls():
    handler = web._Redirects(offline=True)  # pylint: disable=protected-access
    with pytest.raises(UsageError):
        handler.redirect_request(None, None, 302, "Found", {}, "file:///etc/passwd")
    with pytest.raises(UsageError):
        handler.redirect_request(None, None, 302, "Found", {}, "https://203.0.113.9/")


def test_fetcher_checks_urls_before_fetching():
    fake = FakeWeb({})
    with pytest.raises(UsageError):
        Fetcher(offline=True, getter=fake).posts("https://203.0.113.9/")
    assert not fake.asked


# -- caching and training ---------------------------------------------------------


def test_save_posts_is_private_and_stable(tmp_path: Path):
    posts = [Post("https://b/1", "Title", "Body."), Post("https://b/2", "", "Other.")]
    saved = save_posts(posts, tmp_path / "web")
    again = save_posts(posts, tmp_path / "web")
    assert saved == again and len(saved) == 2
    first = next(p for p, u in saved.items() if u == "https://b/1")
    assert first.read_text("utf-8") == "Title\n\nBody.\n"
    assert first.stat().st_mode & 0o777 == 0o600


@pytest.fixture
def blog(monkeypatch):
    fake = FakeWeb(
        {
            HOME: page(HOME, "", link=f'<{API}>; rel="https://api.w.org/"'),
            wp_url(1): page(wp_url(1), wp_posts(0, 3)),
        }
    )
    monkeypatch.setattr(web, "http_getter", lambda offline: fake)
    return fake


def test_train_from_a_blog_and_append_offline(blog, registry, corpus):
    code, _, err = run(["train", "--voice", "Blog", HOME], registry=registry)
    assert code == EXIT_OK, err
    assert "read 3 post(s)" in err
    voice = registry.get("Blog")
    assert voice.corpus_summary["documents"] == 3
    assert {s["url"] for s in voice.sources} == {
        f"https://blog.example/post-{i}/" for i in range(3)
    }
    for source in voice.sources:
        assert Path(source["path"]).parent == registry.path_for("Blog") / web.WEB_DIR

    # Appending local files re-measures the cached posts without the network.
    blog.pages.clear()
    code, _, err = run(["train", "--voice", "Blog", "--append", str(corpus)], registry=registry)
    assert code == EXIT_OK, err
    voice = registry.get("Blog")
    assert sum("url" in s for s in voice.sources) == 3
    assert len(voice.sources) > 3


def test_train_reports_unreachable_and_empty_sites(blog, registry, corpus):
    code, _, err = run(["train", "--voice", "X", "https://gone.example/"], registry=registry)
    assert code == EXIT_FETCH and "HTTP 404" in err
    blog.pages["https://empty.example/"] = page("https://empty.example/", "<p></p>")
    code, _, err = run(
        ["train", "--voice", "X", str(corpus), "https://empty.example/"], registry=registry
    )
    assert code == EXIT_OK and "no writing found" in err


def test_train_offline_refuses_remote_urls(blog, registry):
    code, _, err = run(["train", "--voice", "X", "--offline", HOME], registry=registry)
    assert code == EXIT_USAGE and "offline" in err
    assert not blog.asked


def test_train_honors_config_and_no_config(blog, registry, tmp_path, monkeypatch):
    rc = tmp_path / "rc"
    rc.write_text("[backend]\noffline = on\n", "utf-8")
    rc.chmod(0o600)
    monkeypatch.setenv("PROSELIGN_CONFIG", str(rc))
    code, _, err = run(["train", "--voice", "X", HOME], registry=registry)
    assert code == EXIT_USAGE and "offline" in err
    other = tmp_path / "other"
    other.write_text("[backend]\noffline = off\n", "utf-8")
    other.chmod(0o600)
    code, _, err = run(["train", "--voice", "X", "--config", str(other), HOME], registry=registry)
    assert code == EXIT_OK, err
    # Settings options may come before the subcommand, too.
    code, _, err = run(["--no-config", "train", "--voice", "Y", HOME], registry=registry)
    assert code == EXIT_OK, err
    code, _, err = run(["--config", str(other), "train", "--voice", "Z", HOME], registry=registry)
    assert code == EXIT_OK, err
    code, _, err = run([f"--config={other}", "train", "--voice", "W", HOME], registry=registry)
    assert code == EXIT_OK, err


def test_train_refuses_a_symlinked_voice_directory(blog, registry, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    registry.root.mkdir(parents=True)
    registry.path_for("Linked").symlink_to(elsewhere)
    code, _, err = run(["train", "--voice", "Linked", HOME], registry=registry)
    assert code != EXIT_OK and "symbolic link" in err
    assert not list(elsewhere.iterdir())
