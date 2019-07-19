##############################################
# Note that these global variables need to be set properly
#  to refer to the local installation of Tree-tagger
# See https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
#  for how to download

TREETAG = "C:\\TreeTagger\\"
TREETAGCOMMAND = "bin\\tag-english"

##############################################

import re
import tempfile

##############################################

def treeTag(txt, encTo="UTF-8", command=TREETAG+TREETAGCOMMAND):
    txt = txt.encode('latin-1','replace').decode('latin-1')
    txt = re.sub("^[ \t\r\n\v\f]*","",txt)
    txt = re.sub("[ \t\r\n\v\f]*$","",txt)

    tmpdir = tempfile._get_default_tempdir()
    ff = tmpdir +'\\' + next(tempfile._get_candidate_names())
    fo = tmpdir +'\\' + next(tempfile._get_candidate_names())

    f = open(ff,"w+")
    f.write(txt)
    f.close()
    cmd = command + ' '  + ff + ' > ' + fo
    os.system(cmd)

    f = open(fo,'r')
    s = f.read().split('\n')
    wrd = []
    tag = []
    stm = []
    for ss in s:
        ssp = ss.split('\t')
        if len(ssp)==3:
            wrd.append(ssp[0])
            tag.append(ssp[1])
            stm.append(ssp[2])
        
    o = pd.DataFrame()
    o['tag'] = tag
    o['stem'] = stm
    o['word'] = wrd

    return(o)
    
def treeTagAllText(txtlist,encTo="UTF-8",delim=" \x1B@\n"):
    clps = ''
    for s in txtlist:
        clps = clps + delim + s

    delimb = re.sub("^ ", "", re.sub("\n","",delim))
    taggedText = treeTag(clps, encTo=encTo)

    id = (taggedText['word'] == delimb).cumsum()
    g = taggedText.groupby(id)
    
    out = []
    for k in g.groups.keys():
        out.append(g.get_group(k)[1:])

    return(out)