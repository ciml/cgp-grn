import pandas as pd
import numpy as np
from csaps import csaps
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from os.path import exists
import gc

class Pseudotime:
    def __init__(self, n_pseudotimes, pseudotime_file, pseudotimes = [], ordered_pseudotimes = []):
        self.pseudotime_file = pseudotime_file
        self.n_pseudotimes = n_pseudotimes
        self.pseudotimes = pseudotimes
        self.ordered_pseudotimes = ordered_pseudotimes
        self.pseudotimesToProcess = []
        self.cells = []
        for n_pseudotime in range(n_pseudotimes):
            pseudotimes.append({})
            ordered_pseudotimes.append([])
    
    def processPseudotime(self, pseudotime):
        allCells = list(self.pseudotime_file.T.columns)
        self.cells = allCells
        for cell in allCells:
            currentPseudotime = self.pseudotime_file.T[cell][pseudotime]
            if currentPseudotime not in self.pseudotimes[pseudotime].keys():
                self.pseudotimes[pseudotime][currentPseudotime] = [cell]
            else:
                self.pseudotimes[pseudotime][currentPseudotime].append(cell)
        allPseudotimes = list(self.pseudotimes[pseudotime].keys())
        self.ordered_pseudotimes = list(self.pseudotimes[pseudotime].keys())
        self.ordered_pseudotimes.sort()
        for pt in allPseudotimes:
            if len(self.pseudotimes[pseudotime][pt]) > 1:
                self.pseudotimesToProcess.append(pt)


class ExpressionData:
    def __init__(self, n_genes, expression_file, pseudotime_object, genes=[], cells=[]):
        self.n_genes = n_genes
        self.expression_file = expression_file
        self.genes = genes
        self.cells = cells
        self.pseudotime_object = pseudotime_object
        self.processedPTs = {}

    def processPTs(self, n_pseudotime):
        for ptValue in self.pseudotime_object.pseudotimesToProcess:
            cellsToProcess = self.pseudotime_object.pseudotimes[n_pseudotime][ptValue]
            processedCell = []
            for cell in cellsToProcess:
                if cell == cellsToProcess[0]:
                    processedCell = list(self.expression_file[cell])
                else:
                    currentCellExpressionData = list(self.expression_file[cell])
                    for i in range(len(currentCellExpressionData)):
                        if currentCellExpressionData[i] != 0.0:
                            processedCell[i] = (processedCell[i] + currentCellExpressionData[i])/2
            self.processedPTs[ptValue] = processedCell        


    def calculateSmooth(self, expressionData, timePoints):
        #smooth_genes = {}
        rs = ShuffleSplit(n_splits = 10, train_size = 0.1, test_size = 0.9, random_state=1345540)
        smooth_errors = {}
        for train, test in rs.split(expressionData):
            train.sort()
            test.sort()
            vector_train = [[],[]]
            vector_test = [[],[]]
            for element in train:
                vector_train[0].append(timePoints[element])
                vector_train[1].append(expressionData[element])
            for element in test:
                vector_test[0].append(timePoints[element])
                vector_test[1].append(expressionData[element])

            smooth_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
            for smooths in smooth_values:
                ys = csaps(vector_train[0], vector_train[1], timePoints, smooth=smooths)
                dif_total = 0
                for element in range(len(vector_test[0])):
                    calculated = ys[element]
                    real = vector_test[1][element]
                    dif_total += pow(real - calculated, 2)
                if smooths not in smooth_errors.keys():
                    smooth_errors[smooths] = dif_total
                else:
                    smooth_errors[smooths] += dif_total

            best_smooth = list(smooth_errors.keys())[np.argmin(list(smooth_errors.values()))]

            #smooth_genes[gene_name] = best_smooth
        return best_smooth
    
    def generateSplineCurve(self, expressionData, timePoints, smoothValue, timePointsClean, geneIndex):
        currentGeneName = self.expression_file.T.columns[geneIndex]
        spline_expressionData = []
        ys = csaps(timePointsClean, expressionData, timePoints, smooth=smoothValue)
        #plt.scatter(timePointsClean, expressionData)
        #plt.plot(timePoints, ys, c='black')
        #plt.show()
        #print(ys)
        #print(len(ys))
        currentDict = {}
        currentDict[currentGeneName] = ys
        df = pd.DataFrame(data = currentDict, index=timePoints)
        df.T.to_csv('splineData.csv', mode='a')
        del df
        gc.collect()


        

    def processData(self, n_pseudotime, geneIndex):
        currentGeneName = self.expression_file.T.columns[geneIndex]
        indexesToRemove = []
        currentCellExpressionData = []
        for i in range(len(self.pseudotime_object.ordered_pseudotimes)):
            currentCells = self.pseudotime_object.pseudotimes[n_pseudotime][self.pseudotime_object.ordered_pseudotimes[i]]
            if len(currentCells) == 1:
                currentED = self.expression_file[currentCells[0]][geneIndex]
                if currentED == 0:
                    indexesToRemove.append(i)
                else:
                    currentCellExpressionData.append(self.expression_file[currentCells[0]][geneIndex])
                
            else:
                ptValue = self.pseudotime_object.ordered_pseudotimes[i]
                currentED = self.processedPTs[ptValue][geneIndex]
                if currentED == 0:
                    for k in range(len(self.pseudotime_object.ordered_pseudotimes)):
                        if self.pseudotime_object.ordered_pseudotimes[k] == ptValue:
                            indexesToRemove.append(k)
                else:             
                    currentCellExpressionData.append(self.processedPTs[ptValue][geneIndex])
        
        currentTimePoints = []
        for i in range(len(self.pseudotime_object.ordered_pseudotimes)):
            if i not in indexesToRemove:
                currentTimePoints.append(self.pseudotime_object.ordered_pseudotimes[i])
                
        #best_smooth = self.calculateSmooth(currentCellExpressionData, self.pseudotime_object.ordered_pseudotimes)
        if len(currentTimePoints) < 2:
            print("Not enought expression values for gene: ", currentGeneName) ##aqui colocar > 1
        else:                
            best_smooth = self.calculateSmooth(currentCellExpressionData, currentTimePoints)
            self.generateSplineCurve(currentCellExpressionData, self.pseudotime_object.ordered_pseudotimes, best_smooth, currentTimePoints, geneIndex)
        #print(best_smooth)
        gc.collect()
       

print("Initializing...")
print("Reading pseudotime file...")
pt_File = pd.read_csv('PseudoTime.csv', index_col=0)
pseudotime = Pseudotime(1, pt_File)
print("Processing pseudotimes...")
pseudotime.processPseudotime(0)

#store = pd.HDFStore('teste_final.h5')

print("Reading expression data file...")
#ExprFile = pd.read_csv('ExpressionData_teste.csv', index_col=0)
ExprFileT = pd.read_hdf('train_cite_inputs.h5', start=0, index_col=0)
ExprFile = ExprFileT.T

ExprData = ExpressionData(len(ExprFile.T.columns), ExprFile, pseudotime)

print("Processing expression data...")
ExprData.processPTs(0)
currentPercentage = -1
for gene in range(len(ExprFile.T.columns)):
    printString = str(gene + 1) + " / " + str(len(ExprFile.T.columns))
    print(printString)
    division = round((gene/len(ExprFile.T.columns)) * 100, 2)
    if division != currentPercentage:
        currentPercentage = division
        print(str(currentPercentage) + "%")
    ExprData.processData(0, gene)
    if gene == len(ExprFile.T.columns) - 1:
        print("100%")
