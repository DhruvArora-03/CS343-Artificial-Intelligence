#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

