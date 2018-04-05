#-*- coding:utf-8 -*-

import random
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

class ga:
    def __init__(self,model_obj,score_func,model_params,population,generation,mutation_g,mutation_p,seed,X,y):
        #
        self.model_obj = model_obj
        self.score_func = score_func
        self.model_params = model_params
        #
        self.population = population
        self.generation = generation
        self.mutation_g = mutation_g
        self.mutation_p = mutation_p
        self.nbinary = X.shape[1]
        self.nmax = self.nbinary + self.count_params(model_params)
        self.seed = seed
        #
        self.genes = []
        self.fit = []
        self.genes_child = []
        self.fit_chid = []
        self.elite = [0 for i in range(self.nmax)]
        self.maxfit = -sys.maxint-1
        self.elite_index = -1
        #
        self.expl = X
        self.resp = y
        #
        self.fit_dic = {}
    def count_params(self,model_params):
        count = 0
        for key in model_params.keys():
            if isinstance(self.model_params[key],int) or isinstance(self.model_params[key],float) or isinstance(self.model_params[key],str):
                pass
            elif isinstance(model_params[key][0],int) and isinstance(model_params[key][1],int):
                count += 1
            elif isinstance(model_params[key][0],float) and isinstance(model_params[key][1],float):
                count += 1
        return count
    def int2range(self,ti,rand):
        return int((rand*float(ti[1]-ti[0]+1))+ti[0])
    def float2range(self,tf,rand):
        return rand*(tf[1]-tf[0])
    def set_param_range(self,val):
        if isinstance(val[0],int) and isinstance(val[1],int):
            return self.int2range
        elif isinstance(val[0],float) and isinstance(val[1],float):
            return self.float2range
        else:
            raise TypeError('params is not defined.')
    def gene2cols(self,gene):
        cols = []
        for i in range(len(gene)):
            if gene[i] == 1:
                cols += [i]
        return cols
    def gene2params(self,gene):
        params = {}
        count = 0
        for key in self.model_params.keys():
            if isinstance(self.model_params[key],int) or isinstance(self.model_params[key],float) or isinstance(self.model_params[key],str):
                params[key] = self.model_params[key]
            else:
                params[key] = self.set_param_range(self.model_params[key])(self.model_params[key],gene[count])
                count += 1
        return params
    def gene2fit(self,gene,resp,expl):
        cols = self.gene2cols(gene[0:self.nbinary])
        params = self.gene2params(gene[self.nbinary:self.nmax])
        model_obj = self.model_obj
        for key,val in params.items():
            model_obj.__setattr__(key,val)
        return self.score_func(model=model_obj,X=self.expl[:,cols],y=self.resp)
    def fits(self,genes,flag=True):
        fits = []
        elite = []
        elite_index = -1
        for i, gene in enumerate(genes):
            if tuple(gene) in self.fit_dic:
                tfit = self.fit_dic[tuple(gene)]
            else:
                tfit = self.gene2fit(gene,self.resp,self.expl)
                self.fit_dic[tuple(gene)] = tfit
            fits +=[tfit]
            elite_index = i
        if flag:
            self.elite_index = elite_index
            self.elite = genes[elite_index]
            self.maxfit = fits[elite_index]
        return fits
    def randomkey_gene(self):
        random.seed(self.seed)
        self.genes = [[random.randint(0,1) for j in range(self.nbinary)]+
            [random.random() for j in range(self.nmax - self.nbinary)]
             for i in range(self.population)]
        self.genes_child = [[random.randint(0,1) for j in range(self.nbinary)]+
            [random.random() for j in range(self.nmax - self.nbinary)]
             for i in range(self.population)]
        self.fit = self.fits(self.genes)
    def mutation_onepoint_randomkey(self):
        for i in range(self.population):
            if random.random() < self.mutation_g and i != self.elite_index:
                for point in range(0,self.nbinary):
                    if random.random() < self.mutation_p:
                        #self.genes[i][point] = random.randint(0,1)
                        self.genes[i][point] = 1 - self.genes[i][point]
                for point in range(self.nbinary,self.nmax):
                    if random.random() < self.mutation_p:
                        self.genes[i][point] = random.random()                            
                self.fit[i] = self.gene2fit(self.genes[i],self.resp,self.expl)
    def crossover_uniform_randomkeyER(self):
        family = range(self.population)
        random.shuffle(family)
        self.family = family
        for i in range(self.population/2):
            for j in range(self.nmax):
                r0 = random.randint(0,1)
                self.genes_child[family[2*i]][j] = self.genes[family[2*i+r0]][j]
                self.genes_child[family[2*i+1]][j] = self.genes[family[2*i+1-r0]][j]
        self.fit_child = self.fits(self.genes_child,flag=False)
    def selectionER(self):
        for n in range(len(self.family)/2):
            rank = [1,1,1,1]
            fit_rank = [self.fit[self.family[2*n]],
                self.fit[self.family[2*n+1]],
                self.fit_child[self.family[2*n]],
                self.fit_child[self.family[2*n+1]]]
            for i in range(4):
                for j in range(i+i,4):
                    if fit_rank[i] >= fit_rank[j]:
                        rank[j] += 1
                    if fit_rank[i] < fit_rank[j]:
                        rank[i] += 1
                if rank[0] >= 3 and rank[1] >= 3:
                    self.genes[self.family[2*n]] = list(self.genes_child[self.family[2*n]])
                    self.genes[self.family[2*n+1]] = list(self.genes_child[self.family[2*n+1]])
                    self.fit[self.family[2*n]] = self.fit[self.family[2*n]]
                    self.fit[self.family[2*n+1]] = self.fit[self.family[2*n+1]]
                else:
                    if rank[0] >= 3:
                        if rank[2] <= 2:
                            self.genes[self.family[2*n]] = list(self.genes_child[self.family[2*n]])
                            self.fit[self.family[2*n]] = self.fit[self.family[2*n]]
                        if rank[3] <= 2:
                            self.genes[self.family[2*n]] = list(self.genes_child[self.family[2*n+1]])
                            self.fit[self.family[2*n]] = self.fit[self.family[2*n+1]]
                    if rank[1] >= 3:
                        if rank[2] <= 2:
                            self.genes[self.family[2*n+1]] = list(self.genes_child[self.family[2*n]])
                            self.fit[self.family[2*n+1]] = self.fit[self.family[2*n]]
                        if rank[3] <= 2:
                            self.genes[self.family[2*n+1]] = list(self.genes_child[self.family[2*n+1]])
                            self.fit[self.family[2*n+1]] = self.fit[self.family[2*n+1]]
        for i in range(self.population):
            print "self.fit : "+str(self.fit[i]) #add 20151027
            if self.maxfit < self.fit[i]:
                self.maxfit = self.fit[i]
                self.elite = list(self.genes[i])
                self.elite_index = i
    def main(self):
        ga_res_list = []
        self.randomkey_gene()
        for i in range(self.generation):
            self.crossover_uniform_randomkeyER()
            self.selectionER()
            self.mutation_onepoint_randomkey()
            print '=== generation : {} ==='.format(str(i))
            print 'score : {}'.format(self.maxfit)
            print 'gene : {}'.format(self.elite)
            print 'cols : {}'.format(self.gene2cols(self.elite[0:self.nbinary]))
            print 'params : {}'.format(self.gene2params(self.elite[self.nbinary:self.nmax]))
        ga_res_list += [[self.maxfit,self.elite,self.gene2cols(self.elite[0:self.nbinary]),self.gene2params(self.elite[self.nbinary:self.nmax])]]
        return ga_res_list[len(ga_res_list)-1]



if __name__=='__main__':
    pass
