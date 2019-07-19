#!/usr/bin/env python3

#####################################
# Load critical initial modules
import sys, os

#####################################
# Read configuration variables
with open('ipynb_configs.txt') as fd:
    configs = fd.read()
myvars = [s.split('=') for s in configs.split('\n')]
for v in myvars:
    os.environ[v[0]] = v[1]

#####################################
# Set global archive path
MYARCHV = os.getenv('reddit_archive')

#####################################
# Get parameters

# Subreddit to download
the_subreddit = sys.argv[1]  

# Number of new posts from which to grab comments
if len(sys.argv)>2:
   num_post = int(sys.argv[2])
else:
   num_post = 3

#####################################
# Load utility functions and modules
with open(os.getenv('python_utilities') + 'reddit_utils.py') as fd:
    exec(fd.read())

##################################
# Create and report file names, etc.

# Create timestamp for files
TIMESTAMP = str(dt.datetime.now()).replace(':','-').replace(' ','-').replace('.','-')

# Create file names
fncomms = 'reddit-' + the_subreddit + '-' + TIMESTAMP + '-comments.pkl'
fntags = 'reddit-' + the_subreddit + '-' + TIMESTAMP + '-tags.pkl'

# Report
print('%i posts will be collected' % (num_post))
print('archive location will be %s' % (MYARCHV))
print('comment file will be %s' % (fncomms))
print('tag file will be %s' % (fntags))

##################################
# Collect submissions and comments

# Collect submissions
subs = eah_submission_collector(the_subreddit,num_post)

# This gets the full submission, including the comment forest
#  Could take a long time
uniqSubsFull = eah_get_fullSubmission(subs.ids())

# Get formatted comments
commDB = eah_get_comments(uniqSubsFull)
commDB.to_pickle(MYARCHV + fncomms)

# Clean comments
myParsedComments = eah_parse_comments(commDB)
print('%i comments collected' % (len(myParsedComments['clean_text'])))

##################################
# Use Tree-tagger to conduct part-of-speech tagging
# Note: this could also be done with NLTK module, but
# I am using my own installation of tree-tagger
# See https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/

#  Could take a bit of time
myTaggedText = eah_tag_comments(myParsedComments)
print('%i tags obtained' % myTaggedText.shape[0])

# Save tags
myTaggedText.to_pickle(MYARCHV + fntags)




