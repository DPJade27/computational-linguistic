# -*- coding: utf-8 -*-

#Progetto di Linguistica Computazionale - A.A 2021/2022
# Programma 2
# Giada De Paolis - 615012


import sys
import nltk
import codecs
import re
import math
from nltk import FreqDist
from nltk import trigrams
from nltk import bigrams
import numpy as np


def AnalisiTesto(frasi):
    listaToken = []
    tokenPOStot = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokenPOS = nltk.pos_tag(tokens)
        listaToken += tokens
        tokenPOStot += tokenPOS
    lunghezza = len(listaToken)
    return listaToken, tokenPOStot, lunghezza


#funzione che calcola i 10 pos più frequenti
def POSfreq(tokenPOStot):
    PartOfSpeech = []
    for token in tokenPOStot:
        PartOfSpeech.append(token[1])
    DistribuzionePOS = nltk.FreqDist(PartOfSpeech)
    POSfrequenti = DistribuzionePOS.most_common(10)
    return POSfrequenti


#funzione che calcola i 10 bigrammi più frequenti
def BigrammaFreq(tokenPOStot):
    bigrFreq = []
    bigrammiPOS = list(bigrams(tokenPOStot))
    for bigramma in bigrammiPOS:
        bigrFreq.append(bigramma)
    DistribuzioneBigr = nltk.FreqDist(bigrFreq)
    BigrFrequenti = DistribuzioneBigr.most_common(10)
    return BigrFrequenti


#funzione che calcola i 10 trigrammi più frequenti
def TrigrammaFreq(tokenPOStot):
    trigrFreq = []
    trigrammiPOS = list(trigrams(tokenPOStot))
    for trigramma in trigrammiPOS:
        trigrFreq.append(trigramma)
    DistribuzioneTrigr = nltk.FreqDist(trigrFreq)
    TrigrFrequenti = DistribuzioneTrigr.most_common(10)
    return TrigrFrequenti


#funzione che calcola i 20 aggettivi più frequenti
def AggettiviFreq(tokenPOStot):
    aggFreq = []
    for token in tokenPOStot:
        if token[1] in {"JJ", "JJR", "JJS"}:
            aggFreq.append(token[0])
    DistribuzioneAggettivi = nltk.FreqDist(aggFreq)
    aggettiviFrequenti = DistribuzioneAggettivi.most_common(20)
    return aggettiviFrequenti


#funzione che calcola i 20 avverbi più frequenti
def AvverbiFreq(tokenPOStot):
    avvFreq = []
    for token in tokenPOStot:
        if token[1] in {"RB", "RBR", "RBS"}:
            avvFreq.append(token[0])
    DistribuzioneAvverbi = nltk.FreqDist(avvFreq)
    avverbiFrequenti = DistribuzioneAvverbi.most_common(20)
    return avverbiFrequenti


#estrazione dei 20 bigrammi composti da aggettivo e sostantivo
def BigrammiOrdinati(tokenPOStot, listaToken):
    bigrAS = []
    bigrammiPOS = list(bigrams(tokenPOStot))
    for bigramma in bigrammiPOS:
        if bigramma[0][1] in {"JJ", "JJR", "JJS"}:
           if bigramma[1][1] in {"NN", "NNS", "NNP", "NNPS"}:
               bigrAS.append(bigramma)
    DistribuzioneBigr = nltk.FreqDist(bigrAS)
    listaAggSost = DistribuzioneBigr.most_common(20)
    return listaAggSost
    

def CalcoloBigrammi(tokenPOStot):
    bigrammiPOS = list(bigrams(tokenPOStot))
    return bigrammiPOS

def CalcoloTrigrammi(tokenPOStot):
    trigrammiPOS = list(trigrams(tokenPOStot))
    return trigrammiPOS


#calcolo la frequenza massima
def FreqMax(bigrammiPOS, listaAggSost):
    m = 0
    for bigramma in listaAggSost:
        if bigramma[1] > m:
            m = bigramma[1]
            big = bigramma
    return big


#calcolo la probabilita condizionata massima
def CondMax(bigrammiPOS, listaAggSost, listaToken):
    probCondMax = 0
    for bigramma in listaAggSost:
        freqT1 = listaToken.count(bigramma[0][0][0])
        probCond = bigrammiPOS.count(bigramma[0]) / freqT1
        if(probCond > probCondMax):
            probCondMax = probCond
            big = bigramma
    return (big, "relativa Probabilità Condizionata:",  probCondMax)
    

#calcolo la MI
def MI(listaToken, bigrammiPOS, listaAggSost):
    mi = []
    lunghezzaBigrammi = len(bigrammiPOS)
    m = 0
    for bigramma in listaAggSost:
        frequenzaAggettivo = listaToken.count(bigramma[0][0][0])
        frequenzaSostantivo = listaToken.count(bigramma[0][1][0])
        probabilitaBigr = float(bigramma[1]) / lunghezzaBigrammi
        probAggettivo = float(frequenzaAggettivo) / lunghezzaBigrammi
        probSostantivo = float(frequenzaSostantivo) / lunghezzaBigrammi
        #calcolo Local Mutual Information
        CalcoloProbabilita = probabilitaBigr / (probAggettivo * probSostantivo)
        CalcoloMI = probabilitaBigr * math.log(CalcoloProbabilita ,2)
        if(CalcoloMI>m):
            m = CalcoloMI
            big = bigramma[0]
    return (big[0], "relativo MI:", m)


#estrarre le frasi con almeno 6 token e più corta d 25, dove ogni singolo token occorre almeno 2 volte
def estrazioneToken(listaToken, frasi):
    tokens = []
    frasigiuste = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        if (len(tokens)>6 and len(tokens)<25):
            i=0
            flag=1
            for i in range (len(tokens)):
                freqT=listaToken.count(tokens[i])
                if freqT <2:
                    flag=0
            if flag==1:
                frasigiuste.append(frase)
    return(frasigiuste)

def distrFrequenza(frasi, listaToken):
    distrMin = 100
    distrMax = 0
    distrMedia = 0
    for frase in frasi:
        i = 0
        distribuzione= nltk.FreqDist(nltk.word_tokenize(frase))
        distr = 0
        for tok in distribuzione:
            distr+= distribuzione[tok]
            i+=1
        distr=distr/i
        if distr<distrMin:
            distrMin = distr
            fraseMin=frase
        if distr>distrMax:
            distrMax = distr
            fraseMax = frase
    i=0
    mediaMax=0
    for tok in fraseMax:
        mediaMax+= listaToken.count(tok)
        i+=1
    mediaMax=mediaMax/i

    i=0
    mediaMin=0
    for tok in fraseMin:
        mediaMin+= listaToken.count(tok)
        i+=1
    mediaMin=mediaMin/i
    
    print("media distribuzione più alta:", distrMax, "\ndistribuzione media di frequenza:", mediaMax, "\nla frase:", fraseMax)
    print("media distribuzione più bassa:", distrMin, "\ndistribuzione media di frequenza:", mediaMin, "\nla frase:", fraseMin)


#calcolo Markov del 2° ordine
def Markov(listaToken, frasigiuste):
    bigrammi = CalcoloBigrammi(listaToken)
    trigrammi = CalcoloTrigrammi(listaToken)
    m = 0
    frase = []
    for frasi in frasigiuste:
        bigr = list(bigrams(nltk.word_tokenize(frasi)))
        trigr = list(trigrams(nltk.word_tokenize(frasi)))
        pTrigr = 1
        pBigr = 1
        #probabilita
        for t in trigr:
            pTrigr = pTrigr*trigrammi.count(t)
        pTrigr = pTrigr / len(trigrammi)
        i=0
        for b in bigr:
            if(i>0 or i<len(bigrammi)):
                pBigr = pBigr*bigrammi.count(b)
        pBigr = pBigr / len(bigrammi)
        i+=1
        
        #Markov2
        #print(pTrigr, pBigr)
        probabilita = pTrigr / pBigr
        if probabilita > m:
            m = probabilita
            frase = frasi
    print(frase, "relativa probabilità:", m)
            
    

#calcolo i nomi propri di persona più frequenti con il Named Entity Tagger
def nomiPropriPersona(NamedEntity):
    listaNomi = []
    for nodo in NamedEntity:
        NE = ''
        if hasattr(nodo, 'label'):
            if nodo.label() in ["PERSON"]:
                for partNE in nodo.leaves():
                    NE = NE + partNE[0]
                listaNomi.append(NE)
    DistFrequenza = nltk.FreqDist(listaNomi)
    DistFrequenza = DistFrequenza.most_common(15)
    contatore = 1
    for nome in DistFrequenza:
        print("Il", contatore, "°", "nome proprio di persona più frequente è\"" + nome[0] + "\t", "con frequenza", nome[1])
        contatore = contatore + 1


def main(file1, file2):
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    listaToken1, tokenPOStot1, lunghezza1 = AnalisiTesto(frasi1)
    listaToken2, tokenPOStot2, lunghezza2 = AnalisiTesto(frasi2)

    
    print("**********OUTPUT DEL PROGRAMMA 2**********")
    print()
    print("-----ESTRAZIONE DEI 10 PART-OF-SPEECH PIÙ FREQUENTI-----")
    #calcolo i 10 pos più frequenti
    POSfreq1 = POSfreq(tokenPOStot1)
    print("i 10 pos più frequenti del corpus sull'educazione scolastica in Islanda nel file", file1, "è:", POSfreq1)
    POSfreq2 = POSfreq(tokenPOStot2)
    print("i 10 pos più frequenti del corpus sull'educazione scolastica in Arabia Saudita nel file", file2, "è:", POSfreq2)

    #calcolo i 10 bigrammi più frequenti
    print()
    print("-----ESTRAZIONE DEI 10 BIGRAMMI DI PART-OF-SPEECH PIÙ FREQUENTI-----")
    BigrammaFreq1 = BigrammaFreq(listaToken1)
    print(" i 10 bigrammi più frequnti di", file1, "sono:", BigrammaFreq1)
    BigrammaFreq2 = BigrammaFreq(listaToken2)
    print("i 10 bigrammi più frequenti di", file2, "sono:", BigrammaFreq2)
    
    #calcolo i 10 trigrammi più frequenti
    print()
    print("-----ESTRAZIONE DEI 10 TRIGRAMMI DI PART-OF-SPEECH PIÙ FREQUENTI-----")
    TrigrammaFreq1 = TrigrammaFreq(listaToken1)
    print("i 10 trigrammi più frequenti di", file1, "sono:", TrigrammaFreq1)
    TrigrammaFreq2 = TrigrammaFreq(listaToken2)
    print("i 10 trigrammi più frequenti di", file2, "sono:", TrigrammaFreq2)
    
    #calcolo i 20 aggettivi più frequenti
    print()
    print("-----ESTRAZIONE DEI 20 AGGETTIVI PIÙ FREQUENTI-----")
    AggettiviFreq1 = AggettiviFreq(tokenPOStot1)
    print("i 20 aggettivi più frequenti di", file1, "sono:", AggettiviFreq1)
    AggettiviFreq2 = AggettiviFreq(tokenPOStot2)
    print("i 20 aggettivi più frequenti di", file2, "sono:", AggettiviFreq2)

    #calcolo i 20 avverbi più frequenti
    print()
    print("-----ESTRAZIONE DEI 20 AVVERBI PIÙ FREQUENTI-----")
    AvverbiFreq1 = AvverbiFreq(tokenPOStot1)
    print("i 20 avverbi più frequenti di", file1, "sono:", AvverbiFreq1)
    AvverbiFreq2 = AvverbiFreq(tokenPOStot2)
    print("i 20 avverbi più frequenti di", file2, "sono:", AvverbiFreq2)

    
    listaAggSost1 = BigrammiOrdinati(tokenPOStot1, listaToken1)
    listaAggSost2 = BigrammiOrdinati(tokenPOStot2, listaToken2)

    bigrammiPOS1 = CalcoloBigrammi(tokenPOStot1)
    bigrammiPOS2 = CalcoloBigrammi(tokenPOStot2)

    trigrammiPOS1 = CalcoloTrigrammi(tokenPOStot1)
    trigrammiPOS2 = CalcoloTrigrammi(tokenPOStot2)

    #calcolo frequenza massima
    print()
    print("-----ESTRAZIONE DEI 20 BIGRAMMI COMPOSTI DA AGGETTIVO E SOSTANTIVO----")
    print("-----CALCOLO FREQUENZA MASSIMA CON RELATIVA FREQUENZA-----")
    FreqMax1 = FreqMax(bigrammiPOS1, listaAggSost1)
    print("la frequenza massima del file", file1, "è:", FreqMax1)
    FreqMax2 = FreqMax(bigrammiPOS2, listaAggSost2)
    print("la frequenza massima del file", file2, "è:", FreqMax2)

    
    #calcolo probabilità condizionata massima
    print()
    print("-----CALCOLO PROBABILITÀ CONDIZIONATA MASSIMA CON RELATIVA PROBABILITÀ-----")
    CondMax1 = CondMax(bigrammiPOS1, listaAggSost1, listaToken1)
    print("la probabilità condizionata massima del file", file1, "è:", CondMax1)
    CondMax2 = CondMax(bigrammiPOS2, listaAggSost2, listaToken2)
    print("la probabilità condizionata massima del file", file2, "è:", CondMax2)

    
    #calcolo Local Mutual Information
    print()
    print("-----CALCOLO FORZA ASSOCIATIVA MASSIMA, CALCOLATA IN TERMINI DI LOCAL MUTUAL INFORMATION, CON RELATIVA FORZA ASSOCIATIVA-----")
    MI1 = MI(listaToken1, bigrammiPOS1, listaAggSost1)
    print("la Local Mutual Information del file", file1, "è:", MI1)
    MI2 = MI(listaToken2, bigrammiPOS2, listaAggSost2)
    print("la Local Mutual Information del file", file2, "è:", MI2)


    #estrazione delle frasi con almeno 6 token e più corta di 25 token, dove ogni token ha una frequenza maggiore di 3 
    estrazioneToken1 = estrazioneToken(listaToken1, frasi1)
    estrazioneToken2 = estrazioneToken(listaToken2, frasi2)


    #distribuzione di frequenza
    print()
    print("-----ESTRAZIONE DELLE FRASI CON ALMENO 6 TOKEN E PIÙ CORTA DI 25 TOKEN, DOVE OGNI TOKEN HA UNA FREQUENZA MAGGIORE DI 3-----")
    print("-----CON LA MEDIA DELLA DISTRIBUZIONE DI FREQUENZA PIÙ ALTA, PIÙ BASSA E LA DISTRIBUZIONE MEDIA DI FREQUENZA-----")
    print("di seguito la distribuzione di frequenza del file:", file1)
    distrFrequenza1 = distrFrequenza(estrazioneToken(listaToken1, frasi1),listaToken2)
    print()
    print("di seguito la distribuzione di frequenza del file:", file2)
    distrFrequenza2 = distrFrequenza(estrazioneToken(listaToken2, frasi2), listaToken2)


    #Markov del 2° ordine
    print()
    print("-----CON PROBABILITÀ PIÙ ALTA CALCOLATA ATTRAVERSO UN MODELLO DI MARKOV DI ORDINE 2-----")
    print("di seguito Markov del 2° ordine del file:", file1)
    Markov1 = Markov(listaToken1, estrazioneToken1)
    print()
    print("di seguito Markov del 2° ordine del file:", file2)
    Markov2 = Markov(listaToken2, estrazioneToken2)
    
    
    #calcolo i nomi propri di persona
    NamedEntity1 = nltk.ne_chunk(tokenPOStot1)
    NamedEntity2 = nltk.ne_chunk(tokenPOStot2)
    
    
    print()
    print("-----CLASSIFICAZIONE DELLE ENTITÀ NOMINATE ESTRAENDO I 15 NOMI PROPRI DI PERSONA PIÙ FREQUENTI, ORDINATI PER FREQUENZA-----")
    print("di seguito l'elenco dei primi 15 nomi propri di persona più frequenti del file:", file1)
    nomiPersona1 = nomiPropriPersona(NamedEntity1)
    print()
    print("di seguito l'elenco dei primi 15 nomi propri di persona più frequenti del file:", file2)
    nomiPersona2 = nomiPropriPersona(NamedEntity2)
  


main(sys.argv[1], sys.argv[2])