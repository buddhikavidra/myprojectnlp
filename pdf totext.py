# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:36:19 2020

@author: HP
"""

import pdftotext
filename = 'D:/reserch/law casess/3.pdf'

# Load your PDF
with open("Target.pdf", "rb") as f:
 pdf = pdftotext.PDF(f)

# Save all text to a txt file.
with open('output.txt', 'w') as f:
 f.write("\n\n".join(pdf))