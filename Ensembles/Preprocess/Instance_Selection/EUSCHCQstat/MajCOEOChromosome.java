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
public class MajCOEOChromosome extends  BaseChro {
	/*Useful data for cromosomes*/

	public MajCOEOChromosome (int size) {
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
	public MajCOEOChromosome (MajCOEOChromosome a) {
		int i;
		geneBit = new boolean[a.geneBit.length];
		for (i=0; i<geneBit.length; i++)
			geneBit[i] = a.getGen(i);
		fitness = a.getFit();
		
		valid = true;
		crossed = false;
		prediction = a.prediction.clone();
	}

	/**
	 * Function that evaluates a cromosome
	 * K when evalua its K lables usually be 1
	 * @param elite the elite of cooperation with
	 * @param pFactor boolean
	 * @param P the pFactor
	 */
	public void evalua (
			double datos[][], double real[][], int nominal[][], boolean nulos[][], int clases[],
			double train[][], double trainR[][], int trainN[][], boolean trainM[][], int clasesT[],
			MinCOEOChromosome elite[],
			String wrapper, int K, String evMeas, boolean pFactor, double P,
			int posID, int nPos, int negID, int nNeg,
			boolean distanceEu, keel.Dataset.Attribute entradas[],
			boolean[][] anteriores, boolean[][] salidasAnteriores) {

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
		int nSelectNeg = 0;
		int smoteTotal;
		double tempFitness[];
		prediction = new boolean[datos.length];
		for(i=0; i<nNeg; i++){
			if(geneBit[i] == true)
				nSelectNeg ++;
		}
		while(nSelectNeg==0){
			divergeCHC(0.35, this, 0.25);
			for(i=0; i<nNeg; i++){
				if(geneBit[i] == true)
					nSelectNeg ++;
			}
		}
		
		vecinos = new int[K];
		tempFitness = new double[elite.length];
		nPos = datos.length - nNeg;
		for(k=0; k<elite.length; k++){
			smoteTotal = elite[k].getnSmote();
			s = smoteTotal + nPos + nSelectNeg;
			conjS = new double[s][train[0].length];
			conjR = new double[s][train[0].length];
			conjN = new int[s][train[0].length];
			conjM = new boolean[s][train[0].length];
			clasesS = new int[s];
			h=0;
			for (h=0, l=0; h<nNeg; h++) {
				if (getGen(h)) { 	//selected negtives
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
				for (j=0; j<elite[k].smoteDatosArt[m].length; j++) {
					conjS[l][j] = elite[k].smoteDatosArt[m][j];
					conjR[l][j] = elite[k].smoteRealArt[m][j];
					conjN[l][j] = elite[k].smoteNominalArt[m][j];
					conjM[l][j] = elite[k].smoteNulosArt[m][j];
				}
				clasesS[l] = elite[k].smoteClassS[m];
				l++;
			}
			
			if (wrapper.equalsIgnoreCase("k-NN")) {
				for (i=0; i<datos.length; i++) {
					claseObt = KNN.evaluacionKNN2(K, conjS, conjR, conjN, conjM, clasesS, 
							datos[i], real[i], nominal[i], nulos[i], Math.max(posID, negID) + 1, 
							distanceEu, vecinos);
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
				beta = (double)nSelectNeg /(double)(nPos+smoteTotal);
				if(nSelectNeg==0||(nPos+smoteTotal)==0)
					beta = 0;
				fitness -= Math.abs(1.0-beta) * P;
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
		fitness = 0;
		for(k=0; k<tempFitness.length; k++)
			fitness += tempFitness[k];
		fitness = fitness/k;
		setCrossed(false);
	}
}

