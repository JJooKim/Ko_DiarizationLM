## wer
import string
def wer(ref, hyp):
   #Remove the punctuation from both the truth and transcription
   ref_no_punc = ref.translate(str.maketrans('', '', string.punctuation))
   hyp_no_punc = hyp.translate(str.maketrans('', '', string.punctuation))
   #Calculation starts here
   r = ref_no_punc.split()
   h = hyp_no_punc.split()
   #costs will holds the costs, like in the Levenshtein distance algorithm
   costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
   # backtrace will hold the operations we've done.
   # so we could later backtrace, like the WER algorithm requires us to.
   backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
   OP_OK = 0
   OP_SUB = 1
   OP_INS = 2
   OP_DEL = 3
   DEL_PENALTY = 1
   INS_PENALTY = 1
   SUB_PENALTY = 1
  
   # First column represents the case where we achieve zero
   # hypothesis words by deleting all reference words.
   for i in range(1, len(r)+1):
       costs[i][0] = DEL_PENALTY*i
       backtrace[i][0] = OP_DEL
  
   # First row represents the case where we achieve the hypothesis
   # by inserting all hypothesis words into a zero-length reference.
   for j in range(1, len(h) + 1):
       costs[0][j] = INS_PENALTY * j
       backtrace[0][j] = OP_INS
  
   # computation
   for i in range(1, len(r)+1):
       for j in range(1, len(h)+1):
           if r[i-1] == h[j-1]:
               costs[i][j] = costs[i-1][j-1]
               backtrace[i][j] = OP_OK
           else:
               substitutionCost = costs[i-1][j-1] + SUB_PENALTY
               insertionCost    = costs[i][j-1] + INS_PENALTY
               deletionCost     = costs[i-1][j] + DEL_PENALTY
               
               costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
               if costs[i][j] == substitutionCost:
                   backtrace[i][j] = OP_SUB
               elif costs[i][j] == insertionCost:
                   backtrace[i][j] = OP_INS
               else:
                   backtrace[i][j] = OP_DEL
               
   # back trace though the best route:
   i = len(r)
   j = len(h)
   numSub = 0
   numDel = 0
   numIns = 0
   numCor = 0
   while i > 0 or j > 0:
       if backtrace[i][j] == OP_OK:
           numCor += 1
           i-=1
           j-=1
       elif backtrace[i][j] == OP_SUB:
           numSub +=1
           i-=1
           j-=1
       elif backtrace[i][j] == OP_INS:
           numIns += 1
           j-=1
       elif backtrace[i][j] == OP_DEL:
           numDel += 1
           i-=1
   wer_result = round( ((numSub + numDel + numIns) / (float) (len(r)))*100 , 3)
   #results = {'WER':wer_result, 'numCor':numCor, 'numSub':numSub, 'numIns':numIns, 'numDel':numDel, "numCount": len(r)}
 
   return wer_result

def wder(str1, str2):
    # Split the input strings into lists of integers
    list1 = list(map(int, str1.split()))
    list2 = list(map(int, str2.split()))

    # Get the total number of positions
    total_positions = len(list1)

    # Initialize a counter for non-matching values
    non_matching_count = 0

    # Check each pair of elements at the same position
    for elem1, elem2 in zip(list1, list2):
        if elem1 != elem2:
            non_matching_count += 1

    # Calculate the percentage of non-matching values
    wder = round((non_matching_count / total_positions) * 100, 3)

    return wder

def get_wer(data):
    for utter in data:
        wer = wer(utter['ref_text'], utter['hyp_text'])
        utter['wer'] = wer
        
def get_wder(data):
    for utter in data:
        utter['wder'] = wder(utter['hyp_spk'], utter['hyp_spk_oracle']) 