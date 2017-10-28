package keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.EUSCHCQstat;

import keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.KNN;
import org.core.*;
import keel.Dataset.*;

/**
 * Class to implement a chromosome for the EUS-CHC metho
 * @author Created by Salvador García López (UJA) [19-07-2004]
 * @author Modified by Mikel Galar Idoate (UPNA) [03-05-13]
 * @author Modified by Alberto Fernandez Hilario (UGR) [11-05-16]
 * @version 1.2 (11-05-16)
 * @since JDK 1.5
 */

public class MinCOEOChromosome extends  BaseChro {
	int smoteTotal;
	boolean prediction[];
	double smoteDatosArt[][];
	double smoteRealArt[][];
	int smoteNominalArt[][];
	boolean smoteNulosArt[][];
	int smoteClassS[];
	int kN;
	static int minorCluster[];
	public MinCOEOChromosome (int size) {		
		double u;
		int i;
		geneBit = new boolean[size];
		for (i=0; i<size; i++) {
			u = Randomize.Rand();
			if (u < 0.5) {
				geneBit[i] = false;
			} else {
				geneBit[i] = true;
			}
		}
		crossed = true;
		valid = true;
	}

	/** 
	 * It creates a copied chromosome
	 * @param size
	 * @param a MajCOEOChromosome to copy
	 */
	public MinCOEOChromosome (MinCOEOChromosome a) {
		int i;
		geneBit = new boolean[a.geneBit.length];
		for (i=0; i<geneBit.length; i++)
			geneBit[i] = a.getGen(i);
		fitness = a.getFit();
		valid = true;
		crossed = false;
		prediction = a.prediction.clone();
		
		smoteTotal = a.smoteTotal;
		if(smoteTotal>0){
			int j;
			smoteDatosArt = new double[smoteTotal][a.smoteDatosArt[0].length];
			smoteRealArt = new double[smoteTotal][a.smoteRealArt[0].length];
			smoteNominalArt = new int[smoteTotal][a.smoteNominalArt[0].length];
			smoteNulosArt = new boolean[smoteTotal][a.smoteNulosArt[0].length];
			smoteClassS = new int[smoteTotal];
			for(i=0; i<smoteTotal; i++)
			{
				for(j=0; j<a.smoteDatosArt[i].length; j++)
				{
					smoteDatosArt[i][j] = a.smoteDatosArt[i][j];
					smoteRealArt[i][j] = a.smoteRealArt[i][j];
					smoteNominalArt[i][j] = a.smoteNominalArt[i][j];
					smoteNulosArt[i][j] = a.smoteNulosArt[i][j];
					smoteClassS[i] = a.smoteClassS[i];
				}
			}
		}
	}
	public int getnSmote(){
		return smoteTotal;
	}
	/* smote select positive sample with special nsmote*/
	public int smoteEOU(
			double datos[][], double real[][], int nominal[][], boolean nulos[][], int clases[],
			double train[][], double trainR[][], int trainN[][], boolean trainM[][], int clasesT[],
			//double datosTrain[][], double realTrain[][], int nominalTrain[][], boolean nulosTrain[][], int clasesTrain[],
			int kSMOTE, int smoteN[], int nPos, int posID, int nNeg, int negID, boolean distanceEu,
			 keel.Dataset.Attribute entradas[]){
		int i, j, k, h, m;
		int neighbors[][];
		int nSelPos = 0;
		int posIndex [] = new int[clases.length];
		j = 0;
		for (i = 0; i<clases.length; i++) {
			if (clases[i] == posID) {
				posIndex[i] = j;
				j++;
			}
		}
		for(i=0; i<smoteN.length; i++)
		{
			if(smoteN[i]>0)
				nSelPos ++;
		}
		/* Randomize the instance presentation */
		/*
		for (i = 0; i < positives.length; i++) {
			tmp = positives[i];
			pos = Randomize.Randint(0, positives.length - 1);
			positives[i] = positives[pos];
			positives[pos] = tmp;
		}*/

		/* Obtain k-nearest neighbors of each positive instance */
		neighbors = new int[nSelPos][kSMOTE];
		for (i = 0, j = 0; i < smoteN.length; i++) {
			if(smoteN[i]>0){
				EUSCHCQstat.evaluacionKNNClass(kSMOTE, datos, real, nominal, nulos, clases,
							train[i+nNeg],	trainR[i+nNeg], trainN[i+nNeg], trainM[i+nNeg], 
							Math.max(posID, negID) + 1, distanceEu, neighbors[j], posID);
				j ++;
			}
		}
		int nn;
		int l = 0;
		for(i=0, j=0; i<smoteN.length; i++)
		{
			if(smoteN[i]>0)
			{
				h = 0;
				for(k=0; k<kSMOTE; k++){
					if(neighbors[j][k]!=-1&& minorCluster[posIndex[neighbors[j][k]]] == minorCluster[i]){
						neighbors[j][h] = neighbors[j][k];
						h ++;
					}
				}				
				for(k=0; k<smoteN[i]; k++){
					smoteClassS[l] = posID;
					nn = Randomize.Randint(0, h);
					if(h>0){
						interpola(trainR[i+nNeg],
								real[neighbors[j][nn]],
								trainN[i+nNeg],
								nominal[neighbors[j][nn]],
								trainM[i+nNeg],
								nulos[neighbors[j][nn]], 
							smoteDatosArt[l], smoteRealArt[l], 
							smoteNominalArt[l], smoteNulosArt[l], entradas);
					}else{
						for(m=0; m<trainR[i+nNeg].length; m++){
							smoteDatosArt[l][m] = train[i+nNeg][m];
							smoteRealArt[l][m] = trainR[i+nNeg][m]; 
							smoteNominalArt[l][m] = trainN[i+nNeg][m];
							smoteNulosArt[l][m] = trainM[i+nNeg][m];
						}
						
					}
					l ++;
					
				}
				j ++;
			}
		}
		return l;
	}
	private void interpola(double ra[], double rb[], int na[], int nb[], boolean ma[],
			boolean mb[], double resS[], double resR[], int resN[],
			boolean resM[],  keel.Dataset.Attribute entradas[]) {

		int i;
		double diff;
		double gap;
		int suerte;

		for (i = 0; i < ra.length; i++) {
			if (ma[i] == true && mb[i] == true) {
				resM[i] = true;
				resS[i] = 0;
			} else {
				resM[i] = false;
				if (entradas[i].getType() == Attribute.REAL) {
					diff = rb[i] - ra[i];
					gap = Randomize.Rand();
					resR[i] = ra[i] + gap * diff;
					resS[i] = (ra[i] + entradas[i].getMinAttribute())
							/ (entradas[i].getMaxAttribute() - entradas[i]
									.getMinAttribute());
				} else if (entradas[i].getType() == Attribute.INTEGER) {
					diff = rb[i] - ra[i];
					gap = Randomize.Rand();
					resR[i] = Math.round(ra[i] + gap * diff);
					resS[i] = (ra[i] + entradas[i].getMinAttribute())
							/ (entradas[i].getMaxAttribute() - entradas[i]
									.getMinAttribute());
				} else {
					suerte = Randomize.Randint(0, 2);
					if (suerte == 0) {
						resN[i] = na[i];
					} else {
						resN[i] = nb[i];
					}
					resS[i] = (double) resN[i]
							/ (double) (entradas[i].getNominalValuesList()
									.size() - 1);
				}
			}
		}
	}


	/**
	 * Function that evaluates a cromosome
	 * K when evalua its K lables usually be 1
	 */
	public void evalua (
			double datos[][], double real[][], int nominal[][], boolean nulos[][], int clases[],
			double train[][], double trainR[][], int trainN[][], boolean trainM[][], int clasesT[],
			BaseChro elite[],
			String wrapper, int K, String evMeas, boolean pFactor, double P,
			int posID, int nPos, int negID, int nNeg,
			boolean distanceEu, keel.Dataset.Attribute entradas[],
			boolean[][] anteriores, boolean[][] salidasAnteriores, int bitWidth) {

		int i, j, l=0, m, h, k;
		int aciertosP = 0, aciertosN = 0;
		int totalP = 0, totalN = 0;
		double beta;
		double precision, recall;
		int vecinos[];
		double conjS[][];
		double conjR[][];
		int conjN[][];
		boolean conjM[][];
		int clasesS[];
		int s, claseObt;
		int smoteN[] = new int[nPos];
		int nSelectNeg = 0;
		double tempFitness[];
		prediction = new boolean[datos.length];

		smoteTotal = 0;
		kN = 5;
		boolean binaryCode[] = new boolean [bitWidth];
		for(i=0,j=0; i<geneBit.length; i+=bitWidth,j++){
			binaryCode[0] = geneBit[i];
			for(h=1; h<bitWidth; h++)
				binaryCode[h] = binaryCode[h-1] ^ geneBit[i+h];
			//for(h=1; h<bitWidth; h++)
				//binaryCode[h] = geneBit[i+h];
			
			m = 0;
			int base = 1;
			for(h=bitWidth-1; h>=0; h--){
				if(binaryCode[h] == true)
					m += base;
				base *= 2;
			}
			smoteN[j] = m;
			smoteTotal += m;
		}
		smoteDatosArt = new double[smoteTotal][train[0].length];
		smoteRealArt = new double[smoteTotal][train[0].length];
		smoteNominalArt = new int[smoteTotal][train[0].length];
		smoteNulosArt =  new boolean[smoteTotal][train[0].length];
		smoteClassS = new int[smoteTotal];
		smoteTotal = smoteEOU(
				datos, real, nominal, nulos, clases,
				train, trainR, trainN, trainM, clasesT, 
				kN, smoteN, nPos, posID, nNeg, negID, distanceEu, entradas);
		
		tempFitness = new double[elite.length];
		for(k=0; k<elite.length; k++){
			nSelectNeg = 0;
			for(i=0; i<nNeg; i++){
				if(elite[k].getGen(i) == true)
					nSelectNeg ++;
			}
			nPos = datos.length - nNeg;
			s = smoteTotal + nPos + nSelectNeg;				
			vecinos = new int[K];	        
			conjS = new double[s][train[0].length];
			conjR = new double[s][train[0].length];
			conjN = new int[s][train[0].length];
			conjM = new boolean[s][train[0].length];
			clasesS = new int[s];
			
			h=0;
			for (h=0, l=0; h<nNeg; h++) {
				if (elite[k].getGen(h)) { //selected negtives
					for (j=0; j<train[h].length; j++) {
						conjS[l][j] = train[h][j];
						conjR[l][j] = trainR[h][j];
						conjN[l][j] = trainN[h][j];
						conjM[l][j] = trainM[h][j];
					}
					clasesS[l] = clasesT[h];
					l++;
				}
			}
			//datos[][], double real[][], int nominal[][], boolean nulos[][], int clases[],
			for (m=0; m<datos.length; m++) { // positives
				if (clases[m] == posID) {
					for (j=0; j<datos[m].length; j++) {
						conjS[l][j] = datos[m][j];
						conjR[l][j] = real[m][j];
						conjN[l][j] = nominal[m][j];
						conjM[l][j] = nulos[m][j];
					}
					clasesS[l] = clases[m];
					l++;
				}
			}
			for (m=0; m<smoteTotal; m++) { // positives
				for (j=0; j<smoteDatosArt[m].length; j++) {
					conjS[l][j] = smoteDatosArt[m][j];
					conjR[l][j] = smoteRealArt[m][j];
					conjN[l][j] = smoteNominalArt[m][j];
					conjM[l][j] = smoteNulosArt[m][j];
				}
				clasesS[l] = smoteClassS[m];
				l++;
			}
			if (wrapper.equalsIgnoreCase("k-NN")) {
				for (i=0; i<datos.length; i++) {
					claseObt = KNN.evaluacionKNN2(K, conjS, conjR, conjN, conjM, clasesS, datos[i], real[i], nominal[i], nulos[i], Math.max(posID, negID) + 1, distanceEu, vecinos);
					if (claseObt >= 0)
						if (clases[i] == claseObt && clases[i] == posID) {
							aciertosP++;
							totalP++;
							prediction[i] = true;
						} else if (clases[i] != claseObt && clases[i] == posID) {
							totalP++;
							prediction[i] = false;
						} else if (clases[i] == claseObt && clases[i] != posID) {
							aciertosN++;
							totalN++;
							prediction[i] = true;
						} else if (clases[i] != claseObt && clases[i] != posID) {
							totalN++;
							prediction[i] = false;
						}
				}	    		
			}		    
			
	
			if (evMeas.equalsIgnoreCase("mean")) {
				fitness = Math.sqrt(((double)aciertosP/(double)totalP)*((double)aciertosN/(double)totalN));			
			} else if (evMeas.equalsIgnoreCase("auc")) {
				if (totalP < totalN)
					fitness = (((double)aciertosP / ((double)totalP)) * ((double)aciertosN / ((double)totalN))) + ((1.0 - ((double)aciertosN / ((double)totalN)))*((double)aciertosP / ((double)totalP)))/2.0 + ((1.0 - ((double)aciertosP / ((double)totalP)))*((double)aciertosN / ((double)totalN)))/2.0;
				else
					fitness = (((double)aciertosN / ((double)totalN)) * ((double)aciertosP / ((double)totalP))) + ((1.0 - ((double)aciertosP / ((double)totalP)))*((double)aciertosN / ((double)totalN)))/2.0 + ((1.0 - ((double)aciertosN / ((double)totalN)))*((double)aciertosP / ((double)totalP)))/2.0;			
			} else if (evMeas.equalsIgnoreCase(("cost-sensitive"))) {
				fitness = ((double)totalN - aciertosN) + ((double)totalP - aciertosP) * (double)totalN/(double)totalP;
				fitness /= (2*(double)totalN);
				fitness = 1 - fitness;
			} else if (evMeas.equalsIgnoreCase(("kappa"))) {
				double sumDiagonales = 0.0, sumTrTc = 0.0;
				sumDiagonales = aciertosP + aciertosN;
				sumTrTc = totalP * (totalN - aciertosN) + totalN * (totalP - aciertosP);
				fitness = (((double)datos.length * sumDiagonales - sumTrTc) / ((double)datos.length * (double)datos.length - sumTrTc));
			}
	
			else {
				precision = (((double)aciertosP / ((double)totalP))) / (((double)aciertosP / ((double)totalP)) + (1.0 - ((double)aciertosN / ((double)totalN))));
				recall = (((double)aciertosP / ((double)totalP))) / (((double)aciertosP / ((double)totalP)) + (1.0 - ((double)aciertosP / ((double)totalP))));
				fitness = (2 * precision * recall)/(recall + precision);
			}
	
			if (pFactor) {
				//beta = (double)genes0Activos(clasesT)/(double)genes1Activos(clasesT);				
				beta = (double)nSelectNeg /(double)(nPos+smoteTotal);
				//beta = (double)(nPos+smoteTotal)/(double)nSelectNeg ;
				if(nSelectNeg==0||(nPos+smoteTotal)==0)
					beta = 0;
				fitness -= Math.abs(1.0-beta)*P;
			}
	
			if (anteriores[0] != null) {
				/* Calcular la distancia de Hamming mÃ­nima entre el cromosoma y anteriores[][] */
				double q = -Double.MAX_VALUE;
				for (i = 0; i < anteriores.length && anteriores[i] != null; i++) {
					//double qaux = (1-posRate) * Qstatistic(posa, posCurchrome, posCurchrome.length) + posRate * Qstatistic(nega, negCurchrome, negCurchrome.length);
					//double qaux = (1-posRate) * Hamming(posa, posCurchrome, posCurchrome.length) + posRate * Hamming(nega, negCurchrome, negCurchrome.length);
					//double qaux = (0.5) * Qstatistic(posa, posCurchrome, posCurchrome.length) + 0.5 * Qstatistic(nega, negCurchrome, negCurchrome.length);
					double qaux = Qstatistic(anteriores[i], geneBit, geneBit.length);
					if (q < qaux)
						q = qaux;
				}
				double peso = (double)(anteriores.length - i) / (double) (anteriores.length);
				double IR = (double)totalN / (double)totalP * 0.1;
				fitness = fitness * (1.0 / peso) * (1.0 / IR) - q * peso;
				//fitness = fitness * (1.0 / peso) * (1.0 / IR) + q * peso; //hamming
			}
			if(Double.isNaN(fitness)){
				fitness = Double.MIN_VALUE;
			}
			tempFitness[k] = fitness;
		}
		for(k=0; k<tempFitness.length; k++)
			fitness += tempFitness[k];
		fitness = fitness/k;
		setCrossed(false);
	}
}

