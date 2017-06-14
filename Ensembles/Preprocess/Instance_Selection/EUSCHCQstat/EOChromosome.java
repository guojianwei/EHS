/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. Sánchez (luciano@uniovi.es)
    J. Alcalá-Fdez (jalcala@decsai.ugr.es)
    S. García (sglopez@ujaen.es)
    A. Fernández (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

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
public class EOChromosome implements Comparable {

	/*Cromosome data structure*/
	boolean cuerpo[];

	/*Useful data for cromosomes*/
	double calidad;
	boolean cruzado;
	boolean valido;
	int smoteTotal;
	boolean prediction[];
	double smoteDatosArt[][];
	double smoteRealArt[][];
	int smoteNominalArt[][];
	boolean smoteNulosArt[][];
	int smoteClassS[];
	int kN;
	/** 
	 * Construct a random chromosome of specified size
	 * @param size 
	 */
	public EOChromosome (int size) {

		double u;
		int i;

		cuerpo = new boolean[size];
		for (i=0; i<size; i++) {
			u = Randomize.Rand();
			if (u < 0.5) {
				cuerpo[i] = false;
			} else {
				cuerpo[i] = true;
			}
		}
		cruzado = true;
		valido = true;
		smoteTotal = 0;
	}

	/** 
	 * It creates a copied chromosome
	 * @param size
	 * @param a EOChromosome to copy
	 */
	public EOChromosome (int size, EOChromosome a) {

		int i;
		
		cuerpo = new boolean[size];
		for (i=0; i<cuerpo.length; i++)
			cuerpo[i] = a.getGen(i);
		calidad = a.getCalidad();
		cruzado = false;
		valido = true;
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

	/**
	 * It returns a given gen of the chromsome
	 * @param indice
	 * @return
	 */
	public boolean getGen (int indice) {
		return cuerpo[indice];
	}

	/**
	 * Ite returns the fitness of the chrom.
	 * @return
	 */
	public double getCalidad () {
		return calidad;
	}

	/**
	 * It sets a value for a given chrom.
	 * @param indice
	 * @param valor
	 */
	public void setGen (int indice, boolean valor) {
		cuerpo[indice] = valor;
	} 
	/* smote select positive sample with special nsmote*/
	public int smoteEOU(double datosTrain[][], double realTrain[][], int nominalTrain[][], boolean nulosTrain[][], int clasesTrain[],
			int kSMOTE, int smoteN[], int nPos, int posID, int nNeg, int negID, boolean distanceEu,
			 keel.Dataset.Attribute entradas[], int minorCluster[]){
		int i, j, k, h, m;
		int neighbors[][];
		int nSelPos = 0;
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
				EUSCHCQstat.evaluacionKNNClass(kSMOTE, datosTrain, realTrain, nominalTrain,
							nulosTrain, clasesTrain, datosTrain[i+nNeg],
							realTrain[i+nNeg], nominalTrain[i+nNeg],
							nulosTrain[i+nNeg], Math.max(posID, negID) + 1,
							distanceEu, neighbors[j], posID);
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
					if(neighbors[j][k]!=-1&& (minorCluster[i+nNeg] == minorCluster[neighbors[j][k]])){
						neighbors[j][h] = neighbors[j][k];
						h ++;
					}
				}
				
				for(k=0; k<smoteN[i]; k++){
					smoteClassS[l] = posID;
					nn = Randomize.Randint(0, h);
					if(h>0){
						interpola(realTrain[i+nNeg],
							realTrain[neighbors[j][nn]],
							nominalTrain[i+nNeg],
							nominalTrain[neighbors[j][nn]],
							nulosTrain[i+nNeg],
							nulosTrain[neighbors[j][nn]], 
							smoteDatosArt[l], smoteRealArt[l], 
							smoteNominalArt[l], smoteNulosArt[l], entradas);
					}else{
						for(m=0; m<realTrain[i+nNeg].length; m++){
							smoteDatosArt[l][m] = datosTrain[i+nNeg][m];
							smoteRealArt[l][m] = realTrain[i+nNeg][m]; 
							smoteNominalArt[l][m] = nominalTrain[i+nNeg][m];
							smoteNulosArt[l][m] = nulosTrain[i+nNeg][m];
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
	public void evalua (double datos[][], double real[][], int nominal[][], boolean nulos[][], int clases[], double train[][], double trainR[][], int trainN[][], boolean trainM[][], int clasesT[], String wrapper, int K, String evMeas, boolean MS, boolean pFactor, double P, int posID, int nPos, int negID, int nNeg,boolean distanceEu, keel.Dataset.Attribute entradas[], boolean[][] anteriores, boolean[][] salidasAnteriores, int bitWidth, int minorCluster[], int kN_para) {

		int i, j, l=0, m, h;
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
		prediction = new boolean[train.length];
		for(i=0; i<nNeg; i++){
			if(cuerpo[i] == true)
				nSelectNeg ++;
		}
		while(nSelectNeg==0){
			divergeCHC(0.35, this, 0.25);
			for(i=0; i<nNeg; i++){
				if(cuerpo[i] == true)
					nSelectNeg ++;
			}
		}
		
		smoteTotal = 0;
		kN = kN_para;
		for(i=nNeg,j=0; i<cuerpo.length; i+=bitWidth,j++){
			m = 0;
			int base = 1;
			for(h=bitWidth-1; h>=0; h--){
				if(cuerpo[i+h] == true)
					m += base;
				base *= 2;
			}
			smoteN[j] = m;
			smoteTotal += m;
		}
		if (true) {
			smoteDatosArt = new double[smoteTotal][train[0].length];
			smoteRealArt = new double[smoteTotal][train[0].length];
			smoteNominalArt = new int[smoteTotal][train[0].length];
			smoteNulosArt =  new boolean[smoteTotal][train[0].length];
			smoteClassS = new int[smoteTotal];
			smoteTotal = smoteEOU(train, trainR, trainN, trainM, clasesT, 
					kN, smoteN, nPos, posID, nNeg, negID, distanceEu, entradas, minorCluster);
			s = smoteTotal + nPos + nSelectNeg;				
			vecinos = new int[K];	        
			conjS = new double[s][train[0].length];
			conjR = new double[s][train[0].length];
			conjN = new int[s][train[0].length];
			conjM = new boolean[s][train[0].length];
			clasesS = new int[s];
			
			h=0;
			for (h=0, l=0; h<nNeg; h++) {
				if (getGen(h)) { //selected negtives
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
			for (m=nNeg; m<train.length; m++) { // positives
				for (j=0; j<train[m].length; j++) {
					conjS[l][j] = train[m][j];
					conjR[l][j] = trainR[m][j];
					conjN[l][j] = trainN[m][j];
					conjM[l][j] = trainM[m][j];
				}
				clasesS[l] = clasesT[m];
				l++;
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
		}

		if (evMeas.equalsIgnoreCase("geometric mean")) {
			calidad = Math.sqrt(((double)aciertosP/(double)totalP)*((double)aciertosN/(double)totalN));			
		} else if (evMeas.equalsIgnoreCase("auc")) {
			if (totalP < totalN)
				calidad = (((double)aciertosP / ((double)totalP)) * ((double)aciertosN / ((double)totalN))) + ((1.0 - ((double)aciertosN / ((double)totalN)))*((double)aciertosP / ((double)totalP)))/2.0 + ((1.0 - ((double)aciertosP / ((double)totalP)))*((double)aciertosN / ((double)totalN)))/2.0;
			else
				calidad = (((double)aciertosN / ((double)totalN)) * ((double)aciertosP / ((double)totalP))) + ((1.0 - ((double)aciertosP / ((double)totalP)))*((double)aciertosN / ((double)totalN)))/2.0 + ((1.0 - ((double)aciertosN / ((double)totalN)))*((double)aciertosP / ((double)totalP)))/2.0;			
		} else if (evMeas.equalsIgnoreCase(("cost-sensitive"))) {
			calidad = ((double)totalN - aciertosN) + ((double)totalP - aciertosP) * (double)totalN/(double)totalP;
			calidad /= (2*(double)totalN);
			calidad = 1 - calidad;
		} else if (evMeas.equalsIgnoreCase(("kappa"))) {
			double sumDiagonales = 0.0, sumTrTc = 0.0;
			sumDiagonales = aciertosP + aciertosN;
			sumTrTc = totalP * (totalN - aciertosN) + totalN * (totalP - aciertosP);
			calidad = (((double)datos.length * sumDiagonales - sumTrTc) / ((double)datos.length * (double)datos.length - sumTrTc));
		}

		else {
			precision = (((double)aciertosP / ((double)totalP))) / (((double)aciertosP / ((double)totalP)) + (1.0 - ((double)aciertosN / ((double)totalN))));
			recall = (((double)aciertosP / ((double)totalP))) / (((double)aciertosP / ((double)totalP)) + (1.0 - ((double)aciertosP / ((double)totalP))));
			calidad = (2 * precision * recall)/(recall + precision);
		}

		if (pFactor) {
			if (MS) {
				beta = (double)genesActivos()/(double)nPos;				
			} else {
				//beta = (double)genes0Activos(clasesT)/(double)genes1Activos(clasesT);				
				beta = (double)nSelectNeg /(double)(nPos+smoteTotal);
				//beta = (double)(nPos+smoteTotal)/(double)nSelectNeg ;
				if(nSelectNeg==0||(nPos+smoteTotal)==0)
					beta = 0;
			}
			calidad -= Math.abs(1.0-beta)*P;
		}

		if (anteriores[0] != null) {
			/* Calcular la distancia de Hamming mÃ­nima entre el cromosoma y anteriores[][] */
			boolean negCurchrome[] = new boolean[nNeg];
			boolean posCurchrome[] = new boolean[nPos*bitWidth];
			boolean nega[] = new boolean[nNeg];
			boolean posa[] = new boolean[nPos*bitWidth];
			double posRate = (double)totalP/( (double)totalN + (double)totalP );
			for(i = 0; i < nNeg; i++)
				negCurchrome[i] = cuerpo[i];
			for(i = 0; i < nPos*bitWidth; i++)
				posCurchrome[i] = cuerpo[i+nNeg];

			double q = -Double.MAX_VALUE;
			for (i = 0; i < anteriores.length && anteriores[i] != null; i++) {
				for(j = 0; j < nNeg; j++)
					nega[j] = anteriores[i][j];
				for(j = 0; j < nPos*bitWidth; j++)
					posa[j] = anteriores[i][j+nNeg];
						
				double qaux = (1-posRate) * Qstatistic(posa, posCurchrome, posCurchrome.length) + posRate * Qstatistic(nega, negCurchrome, negCurchrome.length);
				//double qaux = Qstatistic(anteriores[i], cuerpo, clases.length);
				if (q < qaux)
					q = qaux;
			}
			double peso = (double)(anteriores.length - i) / (double) (anteriores.length);
			double IR = (double)totalN / (double)totalP * 0.1;
			calidad = calidad * (1.0 / peso) * (1.0 / IR) - q * peso;
		}
		if(Double.isNaN(calidad)){
			calidad = Double.MIN_VALUE;
		}
		cruzado = false;
	}


	private double Qstatistic(boolean[] v1, boolean[] v2, int n) {
		double[][] t = new double[2][2];
		double ceros = 0;
		if (v1.length < n)
			n = v1.length;
		for (int i = 0; i < n; i++) {
			if (v1[i] == v2[i] && v1[i] == true)
				t[0][0]++;
			else if (v1[i] == v2[i] && v1[i] == false)
				t[1][1]++;
			else if (v1[i] != v2[i] && v1[i] == true)
				t[1][0]++;
			else
				t[0][1]++;
			if (!v2[i])
				ceros++;
		}
		if (ceros == n)
			return 2.0;
		return (t[1][1] * t[0][0] - t[0][1] * t[1][0]) / (t[1][1] * t[0][0] + t[0][1] * t[1][0]);
	}

	/**
	 * Function that does the CHC diverge
	*/
	public void divergeCHC (double r, EOChromosome mejor, double prob) {

		int i;

		for (i=0; i<cuerpo.length; i++) {
			if (Randomize.Rand() < r) {
				if (Randomize.Rand() < prob) {
					cuerpo[i] = true;
				} else {
					cuerpo[i] = false;
				}
			} else {
				cuerpo[i] = mejor.getGen(i);
			}
		}
		cruzado = true;
	}

	public boolean estaEvaluado () {
		return !cruzado;
	}

	public int genesActivos () {

		int i, suma = 0;

		for (i=0; i<cuerpo.length; i++) {
			if (cuerpo[i]) suma++;
		}

		return suma;
	}

	public int genes0Activos (int clases[]) {

		int i, suma = 0;

		for (i=0; i<cuerpo.length; i++) {
			if (cuerpo[i] && clases[i] == 0) suma++;
		}

		return suma;
	}

	public int genes1Activos (int clases[]) {

		int i, suma = 0;

		for (i=0; i<cuerpo.length; i++) {
			if (cuerpo[i] && clases[i] == 1) suma++;
		}

		return suma;
	}

	public boolean esValido () {
		return valido;
	}

	public void borrar () {
		valido = false;
	}

	/**
	 * Function that lets compare cromosomes for an easilier sort
	*/
	public int compareTo (Object o1) {
		if (this.calidad > ((EOChromosome)o1).getCalidad())
			return -1;
		else if (this.calidad < ((EOChromosome)o1).getCalidad())
			return 1;
		else return 0;
	}

	/**
	 * Prints the chrosome into a string value
	 */
	public String toString() {

		int i;
		String temp = "[";

		for (i=0; i<cuerpo.length; i++)
			if (cuerpo[i])
				temp += "1";
			else
				temp += "0";
		temp += ", " + String.valueOf(calidad) + ", " + String.valueOf(genesActivos()) + "]";

		return temp;
	}
}

