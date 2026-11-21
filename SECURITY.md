# Security

Please report a security problem privately, through GitHub's private vulnerability
reporting: on this repository's **Security** tab, choose **Report a vulnerability**. Do not
open a public issue for it.

Say what you found, how to reproduce it, and what it could expose. You will get an answer
when the report has been read; a fix, if one is needed, goes into `main` with a note in its
commit.

Calico is designed to run on one computer for its owner: the web interface answers only
that computer (`calico serve` refuses any other host), and the archive is only ever read.
A way around either of those is exactly what this policy is for.
