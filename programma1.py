# -*- coding: utf-8 -*-

#Progetto di Linguistica Computazionale - A.A 2021/2022
# Programma 1
# Giada De Paolis - 615012


import sys
import nltk
from nltk import pos_tag
    

#funzione che calcola la distribuzione in termini di percentuale delle parole piene e parole funzionali
def AnalisiTesto(frasi):
    lunghezzaTotale = 0.0
    listaToken = []
    tokenPOStot = []
    for frase in frasi:
        token = nltk.word_tokenize(frase)
        tokenPOS = nltk.pos_tag(token)
        listaToken = listaToken + token
        tokenPOStot = tokenPOStot + tokenPOS
        lunghezzaTotale = lunghezzaTotale + len(token)

    return lunghezzaTotale, listaToken, tokenPOStot
    

def CalcoloPercentualeParolePiene(tokenPOStot, frasi):
    numAggettivi = 0
    numSostantivi = 0
    numVerbi = 0
    numAvverbi = 0
    for token in tokenPOStot:
        if token[1] in {"JJ", "JJR", "JJS"}:
            numAggettivi += 1
        if token[1] in {"NN", "NNS", "NNP", "NNPS"}:
            numSostantivi += 1
        if token[1] in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
            numVerbi += 1
        if token[1] in {"RB", "RBR", "RBS"}:
            numAvverbi += 1
    
    #calcolo la percentuale
    percAggettivi = float(numAggettivi/100)
    percSostantivi = float(numSostantivi/100)
    percVerbi = float(numVerbi/100)
    percAvverbi = float(numAvverbi/100)

    return percAggettivi, percSostantivi, percVerbi, percAvverbi


def CalcoloPercentualeParoleFunzionali(tokenPOStot, frasi):
    numArticoli = 0
    numPreposizioni = 0
    numCongiunzioni = 0
    numPronomi = 0
    for token in tokenPOStot:
        if token[1] in {"AT"}:
            numArticoli += 1
        if token[1] in {"IN"}:
            numPreposizioni = 0
        if token[1] in {"CC"}:
            numCongiunzioni += 1
        if token[1] in {"PRP", "PRP$"}:
            numPronomi +=1

    #calcolo la percentuale
    percArticoli = float(numArticoli/100)
    percPreposizioni = float(numPreposizioni/100)
    percCongiunzioni = float(numCongiunzioni/100)
    percPronomi = float(numPronomi/100)

    return percArticoli, percPreposizioni, percCongiunzioni, percPronomi


#funzione che calcola il numero di hapax sui primi 1000 token
def CalcoloHapax(tokens):
    hapax = 0
    vocabolario = set(tokens)
    for token in vocabolario:
    	freqToken = tokens.count(token)
        if freqToken == 1:
    	   hapax += 1
    return hapax


#funzione che calcola il vocabolario e la TTR per porzioni incrementali
def CalcoloIncrementale(tokens, lunghezza):
    for i in range(0, len(tokens), 500):
        tokens500 = tokens[0:i+500]
        vocabolario500 = list(set(tokens500))
        ttr500 =float(len(vocabolario500)) / float(len(tokens500))
        print(i, '-', i+500)
        print("dimensioni del corpus:", len(tokens500))
        print("dimensioni del vocabolario:", len(vocabolario500))
        print("dimensioni della TTR:", ttr500)


#funzione che calcola la lunghezza media dei token in temini di caratteri, punteggiatura esclusa
def CalcoloLunghezzaMediaToken(tokens):
    lunghezzaTOT = 0.0
    lunghezzaCaratteriTOT = 0.0
    tokensNoPunteggiatura = []
    for token in tokens:
        if not(token in[".",",",";",":","!","?","(",")","[","]","-","*","/","'","<",">"]):
            lunghezzaToken = len(token)
            lunghezzaCaratteriTOT+=lunghezzaToken
            lunghezzaTOT = lunghezzaTOT + 1
            lunghezzaMediaToken = float(lunghezzaCaratteriTOT)/lunghezzaTOT*1.0

    return lunghezzaMediaToken

    
#funzione che calcola la lunghezza media delle frasi in termini di token
def CalcoloLunghezza(frasi):
    lunghezzaMedia = 0.0
    tokensTOT = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensTOT+=tokens
        lunghezzaMedia+=len(frase)
    lunghezzaMedia = lunghezzaMedia / len(frasi)

    return lunghezzaMedia, tokensTOT


#funzione che calcola il numero di frasi e di token
def CalcoloNumeroFrasieTokens(frasi):
    numFrasi = 0.0
    numTokens = 0.0
    tokensTOT = []
    for frase in frasi:
        frase = frase.lower()
        tokens = nltk.word_tokenize(frase)
        numFrasi+=1
        numTokens+=len(tokens)
        tokensTOT+=tokens
    numTokens = len(tokensTOT)
        
    return numFrasi, numTokens, tokensTOT


def main(file1, file2):
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    
    #chiamo la funzione CalcoloNumeroFrasieTokens
    numFrasi1, numTokens1, tokens1  = CalcoloNumeroFrasieTokens(frasi1)
    numFrasi2, numTokens2, tokens2  = CalcoloNumeroFrasieTokens(frasi2)

    print("**********OUTPUT DEL PROGRAMMA1**********")
    print()
    print("-----NUMERO DI FRASI E TOKEN-----")
    print("il numero di frasi nel file", file1, "sul sitema educativo in islanda è:", str(numFrasi1), "mentre il numero di token è:", str(numTokens1))
    print()
    print("il numero di frasi nel file", file2, "sul sistema educativo in Arabia Saudita è:", str(numFrasi1), "mentre il numero di token è:",str(numTokens2))

    #chiamo la funzione CalcoloLunghezza
    lunghezzaMedia1, tokens1 = CalcoloLunghezza(frasi1)
    lunghezzaMedia2, tokens2 = CalcoloLunghezza(frasi2)
    
    print()
    print("-----LUNGHEZZA MEDIA DELLE FRASI IN TERMINI DI TOKEN-----")
    print("la lunghezza media delle frasi in termini di token del file:", file1, "è:", lunghezzaMedia1)
    print("la lunghezza media delle frasi in termini di token del file:", file2, "è:", lunghezzaMedia2)

    #chiamo la funzione CalcoloLunghezzaMediaToken
    lunghezzaMediaToken1 = CalcoloLunghezzaMediaToken(tokens1)
    lunghezzaMediaToken2 = CalcoloLunghezzaMediaToken(tokens2)

    print()
    print("-----LUNGHEZZA MEDIA DEI TOKEN IN TERMINI DI CARATTERI-----")
    print("la lunghezza media dei token in termini di caratteri  del file:", file1, "è di:", lunghezzaMediaToken1, "caratteri")
    print("la lunghezza media dei token in termini di caratteri del file:", file2, "è di:", lunghezzaMediaToken2, "caratteri") 
          
    #chiamo la funzione CalcoloHapax
    numHapax1 = CalcoloHapax(tokens1[:1000])
    numHapax2 = CalcoloHapax(tokens2[:1000])
    print()
    print("-----NUMERO DI HAPAX SUI PRIMI 1000 TOKEN-----")
    print("il numero degli hapax sui primi 1000 token del file", file1, "è:", numHapax1)
    print()
    print("il numero degli hapax sui primi 1000 token del file", file2, "è:", numHapax2)

    #chiamo la funzione CalcoloIncrementale
    print()
    print("-----GRANDEZZA DEL VOCABOLARIO E RICCHEZZA LESSICALE ALL'AUMENTARE DEL CORPUS PER PORZIONI INCREMENTALI DI 500 TOKEN-----")
    print("*DI SEGUITO IL CALCOLO INCREMENTALE DEL FILE SULL'EDUCAZIONE IN ISLANDA")
    CalcoloIncrementale(tokens1, len(file1))
    print()
    print("*DI SEGUITO IL CALCOLO INCREMENTALE DEL FILE SULL'EDUCAZIONE IN ARABIA SAUDITA")
    CalcoloIncrementale(tokens2, len(file2))

    #chiamo la funzione CalcoloPercentualeParolePiene
    lunghezza1, listaToken1, tokenPOStot1 = AnalisiTesto(frasi1)
    lunghezza2, listaToken2, tokenPOStot2 = AnalisiTesto(frasi2)
    
    percSostantivi1, percAggettivi1, percVerbi1, percAvverbi1 = CalcoloPercentualeParolePiene(tokenPOStot1, frasi1)
    percSostantivi2, percAggettivi2, percVerbi2, percAvverbi2 = CalcoloPercentualeParolePiene(tokenPOStot2, frasi2)

    print()
    print("-----DISTRIBUZIONE IN TERMINI DI PERCENTUALE DELLE PAROLE PIENE----")
    print("nel file", file1, "\nla percentuale dei sostantivi è:", percSostantivi1, "\nla percentuale degli aggettivi è:", percAggettivi1, "\nla percentuale dei verbi è:", percVerbi1, "\nla percentuale degli avverbi è:", percAvverbi1)
    print()
    print("nel file", file2, "\nla percentuale dei sostantivi è:", percSostantivi2, "\nla percentuale degli aggettivi è:", percAggettivi2, "\nla percentuale dei verbi è:", percVerbi2, "\nla percentuale degli avverbi è:", percAvverbi2)

    #chiamo la funzione CalcoloPercentualeParoleFunzionali
    percArticoli1, percPreposizioni1, percCongiunzioni1, percPronomi1 = CalcoloPercentualeParoleFunzionali(tokenPOStot1, frasi1)
    percArticoli2, percPreposizioni2, percCongiunzioni2, percPronomi2 = CalcoloPercentualeParoleFunzionali(tokenPOStot2, frasi2)

    print()
    print("-----DISTRIBUZIONE IN TERMINI DI PERCENTUALE DELLE PAROLE FUNZIONALI-----")
    print("nel file", file1, "\nla percentuale degli articoli è:", percArticoli1, "\nla percentuale delle preposizioni è:", percPreposizioni1, "\nla percentuale delle congiunzioni è:", percCongiunzioni1, "\nla percentuale dei pronomi è:", percPronomi1)
    print()
    print("nel file", file2, "\nla percentuale degli articoli è:", percArticoli2, "\nla percentuale delle preposizioni è:", percPreposizioni2, "\nla percentuale delle congiunzioni è:", percCongiunzioni2, "\nla percentuale dei pronomi è:", percPronomi2)





 
    

main(sys.argv[1], sys.argv[2])