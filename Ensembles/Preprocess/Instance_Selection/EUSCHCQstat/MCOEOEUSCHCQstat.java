/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. SÃ¡nchez (luciano@uniovi.es)
    J. AlcalÃ¡-Fdez (jalcala@decsai.ugr.es)
    S. GarcÃ­a (sglopez@ujaen.es)
    A. FernÃ¡ndez (alberto.fernandez@ujaen.es)
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

//
//  EUSCHC.java
//
//  Salvador García López
//
//  Created by Salvador García López 21-4-2006.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.EUSCHCQstat;

import keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.OutputIS;

import keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.Metodo;
import keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.KNN;

import java.util.StringTokenizer;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Type;
import java.util.Arrays;
import org.core.*;
import keel.Dataset.*;
import keel.Algorithms.ImbalancedClassification.Ensembles.parseParameters;

/**
 * The auxiliary class for the Qstat computation (diversity of the chrosomomes)
 * @author Created by Mikel Galar Idoate (UPNA) [09-05-12]
 * @version 1.1
 * @since JDK 1.5
 */
public class MCOEOEUSCHCQstat extends Metodo {
	/*important var extends from Metodo
	protected double datosTrain[][];
	protected int clasesTrain[];
	protected double datosTest[][];
	protected int clasesTest[];
	protected boolean nulosTrain[][];
	protected int nominalTrain[][];
	protected double realTrain[][];
	 */

	/* Own parameters of the algorithm */
	private long seed;
	private int popSize;
	private int nEval;
	private double r;
	private double prob0to1Rec;
	private double prob0to1Div;
	private double P;
	private int k;
	private String evMeas;
	private boolean majSelection;
	private boolean pFactor;
	private String hybrid;
	private int kSMOTE;
	private int ASMO;
	private double smoting;
	private boolean balance;
	private String wrapper;
	private boolean anterioresMaj[][], anterioresMinor[][],salidasAnteriores[][];
	private boolean[] bestMaj, bestMinor, bestOutputs;
	private int minorCluster[];
	private int nEvaluation, bitWidth;
	private double cp,p;

	/**
	 * Builder with a script file (configuration file)
	 * @param ficheroScript
	 */
	public MCOEOEUSCHCQstat(String ficheroScript) {
		super(ficheroScript);
	}
	
	public void setAnteriores(boolean[][] anterioresMaj,boolean[][] anterioresMinor) {
		this.anterioresMaj = anterioresMaj;
		this.anterioresMinor = anterioresMinor;
	}

	public void setSalidasAnteriores(boolean[][] anteriores) {
		this.salidasAnteriores = anteriores;
	}

	public boolean[] getBestMaj() {
		return bestMaj;
	}
	public boolean[] getBestMinor() {
		return bestMinor;
	}
	public boolean[] getBestOutputs() {
		return bestOutputs;
	}
	
	private BaseChro[] recombination(BaseChro parent[], BaseChro C[], int d, int[] tamC,int []ev, int indice){
		/* Selection(r) of C(t) from P(t) */
		int i,l, tmp, pos;
		int []baraje = new int[popSize];
		Constructor<? extends BaseChro> tmpClass = null;
		try {
			tmpClass = parent[0].getClass().getConstructor( parent[0].getClass());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		for (i = 0; i < popSize; i++)
			baraje[i] = i;
		for (i = 0; i < popSize; i++) {
			pos = Randomize.Randint(i, popSize - 1);
			tmp = baraje[i];
			baraje[i] = baraje[pos];
			baraje[pos] = tmp;
		}
		for (i = 0; i < popSize; i++){
				try {
					C[i] = tmpClass.newInstance(parent[baraje[i]]);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} //new EOChromosome(poblacion[baraje[i]]);
		}
		/* Structure recombination in C(t) constructing C'(t) */
		tamC[indice] = recombinar(C, d);
		BaseChro[] newPob = new BaseChro[tamC[indice]];
		try{
			for (i = 0, l = 0; i < C.length; i++) {
				if (C[i].isValid()) { // the cromosome must be copied to the
					// new poblation C'(t)
					newPob[l] = tmpClass.newInstance(C[i]);
					l++;
				}
			}
		}catch( Exception e){ e.printStackTrace();}
		return newPob;
	}

	private void competition(BaseChro parent[],BaseChro newPob[], int tamC){
		BaseChro pobTemp[];
		pobTemp = new BaseChro[popSize];
		Constructor<? extends BaseChro> tmpClass = null;
		try {
			tmpClass = parent[0].getClass().getConstructor(parent[0].getClass());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		int i, j, l;
		try{
		for (i = 0, j = 0, l = 0; i < popSize && l < tamC; i++) {
			if (parent[j].getFit() > newPob[l].getFit()) {
				pobTemp[i] = tmpClass.newInstance(parent[j]);
				j++;
			} else {
					pobTemp[i] = tmpClass.newInstance(newPob[l]);
				l++;
			}
		}
		if (l == tamC) { // there are cromosomes for copying
			for (; i < popSize; i++) {
					pobTemp[i] = tmpClass.newInstance(parent[j]);
				j++;
			}
		}
		}catch (Exception e){
			e.printStackTrace();
		}
		//parent = pobTemp;
		for (i = 0; i < popSize; i++ )
			parent[i]= pobTemp[i];
	}
	private int doDiverge(BaseChro poblacion[], int chromeSize){
		int i, d;
		for (i = 1; i < popSize; i++) {
			poblacion[i].divergeCHC(r, poblacion[0], prob0to1Div);
		}
		/* Reinicialization of d value */
		d = (int) (r * (1.0 - r) * (double) chromeSize);
		return d;
	}
	public void runAlgorithm() {
		
		double conjS[][];
		double conjR[][];
		int conjN[][];
		boolean conjM[][];
		int clasesS[];
		double datosArt[][];
		double realArt[][];
		int nominalArt[][];
		boolean nulosArt[][];
		int clasesArt[];
		MajCOEOChromosome[] poblacionMaj, eliteMaj;
		MinCOEOChromosome[] poblacionMinor, eliteMinor;
		BaseChro[] newPobMaj;
		BaseChro[] newPobMinor;
		BaseChro C[];
		
		int i, j, l, h;
		int dmaj, dminor, pos, tmp;
		int nSel = 0, nPos = 0, nNeg = 0, posID, negID;
		int eliteSize = 1, chromeSizeMaj, chromeSizeMinor;
		int[] ev = {0};
		int tamC[]= new int[2];
		long tiempo = System.currentTimeMillis();

		posID = clasesTrain[0];
		negID = -1;
		for (i = 0; i < clasesTrain.length; i++) {
			if (clasesTrain[i] != posID) {
				negID = clasesTrain[i];
				break;
			}
		}
		/* Count of number of positive and negative examples */
		for (i = 0; i < clasesTrain.length; i++) {
			if (clasesTrain[i] == posID)
				nPos++;
			else
				nNeg++;
		}
		if (nPos > nNeg) {
			tmp = nPos;
			nPos = nNeg;
			nNeg = tmp;
			tmp = posID;
			posID = negID;
			negID = tmp;
		}
		// posID, negID, nPos, nNeg //Arts are sample, trains are original
		if (hybrid.equalsIgnoreCase("ehs")) {
			/*
			 * negtive sample as left, positive right*/
			datosArt = new double[datosTrain.length][datosTrain[0].length];
			realArt = new double[datosTrain.length][datosTrain[0].length];
			nominalArt = new int[datosTrain.length][datosTrain[0].length];
			nulosArt = new boolean[datosTrain.length][datosTrain[0].length];
			clasesArt = new int[clasesTrain.length];
			l = 0;
			for (i = 0; i < datosTrain.length; i++) {
				if(clasesTrain[i] == negID){
					for (j = 0; j < datosTrain[i].length; j++) {
						datosArt[l][j] = datosTrain[i][j];
						realArt[l][j] = realTrain[i][j];
						nominalArt[l][j] = nominalTrain[i][j];
						nulosArt[l][j] = nulosTrain[i][j];
					}
					clasesArt[l] = clasesTrain[i];
					l ++;
				}
			}
			for (i = 0; i < datosTrain.length; i++) {
				if(clasesTrain[i] == posID){
					for (j = 0; j < datosTrain[i].length; j++) {
						datosArt[l][j] = datosTrain[i][j];
						realArt[l][j] = realTrain[i][j];
						nominalArt[l][j] = nominalTrain[i][j];
						nulosArt[l][j] = nulosTrain[i][j];
					}
					clasesArt[l] = clasesTrain[i];
					l ++;
				}
			}
		}else{
			datosArt = new double[datosTrain.length][datosTrain[0].length];
			realArt = new double[datosTrain.length][datosTrain[0].length];
			nominalArt = new int[datosTrain.length][datosTrain[0].length];
			nulosArt = new boolean[datosTrain.length][datosTrain[0].length];
			clasesArt = new int[clasesTrain.length];
			for (i = 0; i < datosTrain.length; i++) {
				for (j = 0; j < datosTrain[i].length; j++) {
					datosArt[i][j] = datosTrain[i][j];
					realArt[i][j] = realTrain[i][j];
					nominalArt[i][j] = nominalTrain[i][j];
					nulosArt[i][j] = nulosTrain[i][j];
				}
				clasesArt[i] = clasesTrain[i];
			}
		}
		
		chromeSizeMaj = nNeg;
		chromeSizeMinor = nPos*bitWidth;
		dmaj = chromeSizeMaj / 4;
		dminor = chromeSizeMinor / 4;
		/* Random initialization of the population */
		poblacionMaj = new MajCOEOChromosome[popSize];
		poblacionMinor = new MinCOEOChromosome[popSize];
		eliteMaj = new MajCOEOChromosome[eliteSize];
		eliteMinor = new MinCOEOChromosome[eliteSize];
		for (i = 0; i < popSize; i++){
			poblacionMaj[i] = new MajCOEOChromosome(chromeSizeMaj);
			poblacionMinor[i] = new MinCOEOChromosome(chromeSizeMinor);
		}
		for(i=0; i<eliteSize; i++){
			eliteMaj[i] = new MajCOEOChromosome(chromeSizeMaj);
			eliteMinor[i] = new MinCOEOChromosome(chromeSizeMinor);
			for(j=0; j<eliteMinor[i].geneBit.length; j++){
				eliteMinor[i].setGen(j, false);
			}
		}
		/* Initial evaluation of the population */
		for (i = 0; i < popSize; i++){
			poblacionMaj[i].evalua(
					datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
					datosArt, realArt, nominalArt, nulosArt, clasesArt, eliteMinor,
					wrapper, k, evMeas, pFactor, P, posID, nPos, negID, nNeg,
					distanceEu, entradas, anterioresMaj, salidasAnteriores);
			poblacionMinor[i].evalua(
					datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
					datosArt, realArt, nominalArt, nulosArt, clasesArt, eliteMaj,
					wrapper, k, evMeas, pFactor, P, posID, nPos, negID, nNeg,
					distanceEu, entradas, anterioresMinor, salidasAnteriores,bitWidth);
		}
		for(k=0; k<eliteSize; k++) {
			eliteMaj[k] = new MajCOEOChromosome(poblacionMaj[k]);
			eliteMinor[k] =  new MinCOEOChromosome(poblacionMinor[k]);
		}
		
		/* Until stop condition */
		nEval = nEvaluation;
		while (ev[0] < nEval) {
			C = new MajCOEOChromosome[popSize];
			newPobMaj =  recombination(poblacionMaj, C, dmaj, tamC, ev, 0);
			C = new MinCOEOChromosome[popSize];
			newPobMinor = recombination(poblacionMinor, C, dminor,tamC, ev, 1);

			/* Structure evaluation in C'(t) */
			for (i = 0; i < newPobMaj.length; i++) {
				((MajCOEOChromosome) newPobMaj[i]).evalua(
						datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
						datosArt, realArt, nominalArt, nulosArt, clasesArt, eliteMinor,
						wrapper, k, evMeas, pFactor,P, posID, nPos,negID, nNeg,
						distanceEu, entradas, anterioresMaj, salidasAnteriores);
				ev[0]++;
			}
			for (i = 0; i < newPobMinor.length; i++) {
				((MinCOEOChromosome)newPobMinor[i]).evalua(
						datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
						datosArt, realArt, nominalArt, nulosArt, clasesArt, eliteMaj,
						wrapper, k, evMeas, pFactor,P, posID, nPos, negID, nNeg,
						distanceEu, entradas, anterioresMinor, salidasAnteriores,bitWidth);
				ev[0]++;
			}
			for (i = 0; i < popSize; i++) {
				poblacionMaj[i].evalua(
						datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
						datosArt, realArt, nominalArt, nulosArt, clasesArt, eliteMinor,
						wrapper, k, evMeas, pFactor,P, posID, nPos,negID, nNeg,
						distanceEu, entradas, anterioresMaj, salidasAnteriores);
				poblacionMinor[i].evalua(
						datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
						datosArt, realArt, nominalArt, nulosArt, clasesArt, eliteMaj,
						wrapper, k, evMeas, pFactor,P, posID, nPos, negID, nNeg,
						distanceEu, entradas, anterioresMinor, salidasAnteriores,bitWidth);
				ev[0]+=2;
			}
			

			/* Selection(s) of P(t) from C'(t) and P(t-1) */
			Arrays.sort(poblacionMaj);
			Arrays.sort(poblacionMinor);
			Arrays.sort(newPobMaj);
			Arrays.sort(newPobMinor);

			/*
			 * If the best of C' is worse than the worst of P(t-1), then there
			 * will no changes
			 */
			if(tamC[0] == 0	|| newPobMaj[0].getFit() < poblacionMaj[popSize - 1].getFit()){
				dmaj --; 
			} else{
				competition(poblacionMaj, newPobMaj, tamC[0]);
			}
			if(tamC[1] == 0 || newPobMinor[0].getFit() < poblacionMinor[popSize - 1].getFit()){
				dminor --; 
			} else{
				competition(poblacionMinor, newPobMinor, tamC[1]);
			}

			/* Last step of the algorithm */
			if (dmaj <= 0) {
				dmaj = doDiverge(poblacionMaj, chromeSizeMaj);
				for (i = 0; i < popSize; i++)
					if (!(poblacionMaj[i].hasEval())) {
						poblacionMaj[i].evalua(
								datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
								datosArt, realArt, nominalArt, nulosArt, clasesArt,
								eliteMinor,
								wrapper, k, evMeas,	pFactor, P, posID, nPos, negID, nNeg,
								distanceEu, entradas, anterioresMaj, salidasAnteriores);
						ev[0]++;
					}
			}
			if (dminor <= 0) {
				dminor = doDiverge(poblacionMinor, chromeSizeMinor);
				for (i = 0; i < popSize; i++)
					if (!(poblacionMinor[i].hasEval())) {
						poblacionMinor[i].evalua(
								datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain,
								datosArt, realArt, nominalArt, nulosArt, clasesArt,
								eliteMaj,
								wrapper, k, evMeas,	pFactor, P, posID, nPos, negID, nNeg,
								distanceEu, entradas, anterioresMinor, salidasAnteriores,bitWidth);
						ev[0]++;
					}
			}
			for(k=0; k<eliteSize; k++) {
				eliteMaj[k] = new MajCOEOChromosome(poblacionMaj[k]);
				eliteMinor[k] =  new MinCOEOChromosome(poblacionMinor[k]);
			}
		}

		Arrays.sort(poblacionMaj);
		Arrays.sort(poblacionMinor);
		int Nmaj = 0;				
		for (i = 0; i < nNeg; i++) {
			if (poblacionMaj[0].getGen(i)) { // the instance must be copied to
				Nmaj++;
			}
		}
		nSel = Nmaj + nPos + poblacionMinor[0].getnSmote() ;//real???
		System.out.println( "0000000nSel:"+nSel+" Nmaj:"+Nmaj+" nPos:"+nPos+" smoteTotal:"+eliteMinor[0].smoteTotal);
		
		/* Construction of S set from the best cromosome */
		conjS = new double[nSel][datosArt[0].length];
		conjR = new double[nSel][datosArt[0].length];
		conjN = new int[nSel][datosArt[0].length];
		conjM = new boolean[nSel][datosArt[0].length];
		clasesS = new int[nSel];
		
		for (i = 0, l = 0; i < datosArt.length; i++) {
			if ( (i<poblacionMaj[0].geneBit.length&& poblacionMaj[0].getGen(i)) || clasesArt[i] == posID) { // the instance must be copied to
				// the solution
				for (j = 0; j < datosArt[i].length; j++) {
					conjS[l][j] = datosArt[i][j];
					conjR[l][j] = realArt[i][j];
					conjN[l][j] = nominalArt[i][j];
					conjM[l][j] = nulosArt[i][j];
				}
				clasesS[l] = clasesArt[i];
				l++;
			}
		}
		for (i = 0; i < poblacionMinor[0].getnSmote(); i++) {
			// the solution
			for (j = 0; j < poblacionMinor[0].smoteDatosArt[i].length; j++) {
				conjS[l][j] = poblacionMinor[0].smoteDatosArt[i][j];
				conjR[l][j] = poblacionMinor[0].smoteRealArt[i][j];
				conjN[l][j] = poblacionMinor[0].smoteNominalArt[i][j];
				conjM[l][j] = poblacionMinor[0].smoteNulosArt[i][j];
			}
			clasesS[l] = poblacionMinor[0].smoteClassS[i];
			l++;				
		}
		
		/*
		 * for (i = 0; i < poblacion.length; i++){ for (j = 0; j <
		 * poblacion[0].cuerpo.length; j++){
		 * System.out.print((poblacion[i].cuerpo[j] ? 1 : 0)); }
		 * System.out.println(" Calidad: " + poblacion[i].calidad); }
		 */
		bestMaj = poblacionMaj[0].geneBit.clone();
		bestMinor = poblacionMinor[0].geneBit.clone();
		bestOutputs = poblacionMaj[0].prediction.clone();
		System.out
		.println("QstatEUSCHC " + relation + " "
				+ (double) (System.currentTimeMillis() - tiempo)
				/ 1000.0 + "s");

		OutputIS.escribeSalida(ficheroSalida[0], conjR, conjN, conjM, clasesS,
				entradas, salida, nEntradas, relation);
		// OutputIS.escribeSalida(ficheroSalida[1], test, entradas, salida,
		// nEntradas, relation);
	}
	/**
	 * Given the parent and d CHC return newpro
	 * @param d the threshold of recombination
	 * */
	
	private int recombinar(BaseChro C[], int d) {

		int i, j;
		int distHamming;
		int tamC = 0;
		int n = C[0].geneBit.length;
		for (i = 0; i < C.length / 2; i++) {
			distHamming = 0;
			for (j = 0; j < n; j++)
				if (C[i * 2].getGen(j) != C[i * 2 + 1].getGen(j))
					distHamming++;
			if ((distHamming / 2) > d) {
				for (j = 0; j < n; j++) {
					if ((C[i * 2].getGen(j) != C[i * 2 + 1].getGen(j))
							&& Randomize.Rand() < 0.5) {
						if (C[i * 2].getGen(j))
							C[i * 2].setGen(j, false);
						else if (Randomize.Rand() < prob0to1Rec)
							C[i * 2].setGen(j, true);
						if (C[i * 2 + 1].getGen(j))
							C[i * 2 + 1].setGen(j, false);
						else if (Randomize.Rand() < prob0to1Rec)
							C[i * 2 + 1].setGen(j, true);
					}
				}
				tamC += 2;
			} else {
				C[i * 2].borrar();
				C[i * 2 + 1].borrar();
			}
		}

		return tamC;
	}

	/**
	 * SMOTE preprocessing procedure
	 * @param datosTrain input training dta
	 * @param realTrain actual training data
	 * @param nominalTrain nominal attribute values
	 * @param nulosTrain null values
	 * @param clasesTrain training classes
	 * @param datosArt synthetic instances
	 */
	public void SMOTE(double datosTrain[][], double realTrain[][],
			int nominalTrain[][], boolean nulosTrain[][], int clasesTrain[],
			double datosArt[][], double realArt[][], int nominalArt[][],
			boolean nulosArt[][], int clasesArt[], int kSMOTE, int ASMO,
			double smoting, boolean balance, int nPos, int posID, int nNeg,
			int negID, boolean distanceEu) {

		int i, j, l, m;
		int tmp, pos;
		int positives[];
		int neighbors[][];
		double genS[][];
		double genR[][];
		int genN[][];
		boolean genM[][];
		int clasesGen[];
		int nn;

		/* Localize the positive instances */
		positives = new int[nPos];
		for (i = 0, j = 0; i < clasesTrain.length; i++) {
			if (clasesTrain[i] == posID) {
				positives[j] = i;
				j++;
			}
		}

		/* Randomize the instance presentation */
		for (i = 0; i < positives.length; i++) {
			tmp = positives[i];
			pos = Randomize.Randint(0, positives.length - 1);
			positives[i] = positives[pos];
			positives[pos] = tmp;
		}

		/* Obtain k-nearest neighbors of each positive instance */
		neighbors = new int[positives.length][kSMOTE];
		for (i = 0; i < positives.length; i++) {
			switch (ASMO) {
			case 0:
				KNN.evaluacionKNN2(kSMOTE, datosTrain, realTrain, nominalTrain,
						nulosTrain, clasesTrain, datosTrain[positives[i]],
						realTrain[positives[i]], nominalTrain[positives[i]],
						nulosTrain[positives[i]], Math.max(posID, negID) + 1,
						distanceEu, neighbors[i]);
				break;
			case 1:
				evaluacionKNNClass(kSMOTE, datosTrain, realTrain, nominalTrain,
						nulosTrain, clasesTrain, datosTrain[positives[i]],
						realTrain[positives[i]], nominalTrain[positives[i]],
						nulosTrain[positives[i]], Math.max(posID, negID) + 1,
						distanceEu, neighbors[i], posID);
				break;
			case 2:
				evaluacionKNNClass(kSMOTE, datosTrain, realTrain, nominalTrain,
						nulosTrain, clasesTrain, datosTrain[positives[i]],
						realTrain[positives[i]], nominalTrain[positives[i]],
						nulosTrain[positives[i]], Math.max(posID, negID) + 1,
						distanceEu, neighbors[i], negID);
				break;
			}
		}

		/* Interpolation of the minority instances */
		if (balance) {
			genS = new double[nNeg - nPos][datosTrain[0].length];
			genR = new double[nNeg - nPos][datosTrain[0].length];
			genN = new int[nNeg - nPos][datosTrain[0].length];
			genM = new boolean[nNeg - nPos][datosTrain[0].length];
			clasesGen = new int[nNeg - nPos];
		} else {
			genS = new double[(int) (nPos * smoting)][datosTrain[0].length];
			genR = new double[(int) (nPos * smoting)][datosTrain[0].length];
			genN = new int[(int) (nPos * smoting)][datosTrain[0].length];
			genM = new boolean[(int) (nPos * smoting)][datosTrain[0].length];
			clasesGen = new int[(int) (nPos * smoting)];
		}
		for (i = 0; i < genS.length; i++) {
			clasesGen[i] = posID;
			nn = Randomize.Randint(0, kSMOTE - 1);
			interpola(realTrain[positives[i % positives.length]],
					realTrain[neighbors[i % positives.length][nn]],
					nominalTrain[positives[i % positives.length]],
					nominalTrain[neighbors[i % positives.length][nn]],
					nulosTrain[positives[i % positives.length]],
					nulosTrain[neighbors[i % positives.length][nn]], genS[i],
					genR[i], genN[i], genM[i]);
		}

		for (j = 0; j < datosTrain.length; j++) {
			for (l = 0; l < datosTrain[0].length; l++) {
				datosArt[j][l] = datosTrain[j][l];
				realArt[j][l] = realTrain[j][l];
				nominalArt[j][l] = nominalTrain[j][l];
				nulosArt[j][l] = nulosTrain[j][l];
			}
			clasesArt[j] = clasesTrain[j];
		}
		for (m = 0; j < datosArt.length; j++, m++) {
			for (l = 0; l < datosTrain[0].length; l++) {
				datosArt[j][l] = genS[m][l];
				realArt[j][l] = genR[m][l];
				nominalArt[j][l] = genN[m][l];
				nulosArt[j][l] = genM[m][l];
			}
			clasesArt[j] = clasesGen[m];
		}
	}

	/**
	 * Knn evaluation for classification
	 * @return
	 */
	public static int evaluacionKNNClass(int nvec, double conj[][],
			double real[][], int nominal[][], boolean nulos[][], int clases[],
			double ejemplo[], double ejReal[], int ejNominal[],
			boolean ejNulos[], int nClases, boolean distance, int vecinos[],
			int clase) {

		int i, j, l;
		boolean parar = false;
		int vecinosCercanos[];
		double minDistancias[];
		int votos[];
		double dist;
		int votada, votaciones;

		if (nvec > conj.length)
			nvec = conj.length;

		votos = new int[nClases];
		vecinosCercanos = new int[nvec];
		minDistancias = new double[nvec];
		for (i = 0; i < nvec; i++) {
			vecinosCercanos[i] = -1;
			minDistancias[i] = Double.POSITIVE_INFINITY;
		}

		for (i = 0; i < conj.length; i++) {
			dist = KNN.distancia(conj[i], real[i], nominal[i], nulos[i],
					ejemplo, ejReal, ejNominal, ejNulos, distance);
			if (dist > 0 && clases[i] == clase) {
				parar = false;
				for (j = 0; j < nvec && !parar; j++) {
					if (dist < minDistancias[j]) {
						parar = true;
						for (l = nvec - 1; l >= j + 1; l--) {
							minDistancias[l] = minDistancias[l - 1];
							vecinosCercanos[l] = vecinosCercanos[l - 1];
						}
						minDistancias[j] = dist;
						vecinosCercanos[j] = i;
					}
				}
			}
		}

		for (j = 0; j < nClases; j++) {
			votos[j] = 0;
		}

		for (j = 0; j < nvec; j++) {
			if (vecinosCercanos[j] >= 0)
				votos[clases[vecinosCercanos[j]]]++;
		}

		votada = 0;
		votaciones = votos[0];
		for (j = 1; j < nClases; j++) {
			if (votaciones < votos[j]) {
				votaciones = votos[j];
				votada = j;
			}
		}

		for (i = 0; i < vecinosCercanos.length; i++)
			vecinos[i] = vecinosCercanos[i];

		return votada;
	}

	private void interpola(double ra[], double rb[], int na[], int nb[], boolean ma[],
			boolean mb[], double resS[], double resR[], int resN[],
			boolean resM[]) {

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
	 * It reads the configuration file for performing the EUS-CHC method
	 */
	public void readConfiguration(String ficheroScript) {
		parseParameters param = new parseParameters();
		param.parseConfigurationFile(ficheroScript);
		ficheroTraining = param.getTrainingInputFile();
		ficheroTest = param.getTestInputFile();
		ficheroSalida = new String[2];
		ficheroSalida[0] = param.getTrainingOutputFile();
		ficheroSalida[1] = param.getTestOutputFile();
		int i = 0;
		seed = Long.parseLong(param.getParameter(i++));
		popSize = Integer.parseInt(param.getParameter(i++));
		nEval = Integer.parseInt(param.getParameter(i++));
		r = Double.parseDouble(param.getParameter(i++));
		prob0to1Rec = Double.parseDouble(param.getParameter(i++));
		prob0to1Div = Double.parseDouble(param.getParameter(i++));
		wrapper = param.getParameter(i++);
		k = Integer.parseInt(param.getParameter(i++));
		distanceEu = param.getParameter(i++).equalsIgnoreCase("Euclidean") ? true : false;
		evMeas = param.getParameter(i++);
		if (param.getParameter(i++).equalsIgnoreCase("majority_selection"))
			majSelection = true;
		else
			majSelection = false;
		if (param.getParameter(i++).equalsIgnoreCase("EBUS"))
			pFactor = true;
		else
			pFactor = false;
		P = Double.parseDouble(param.getParameter(i++));
		hybrid = param.getParameter(i++);
		kSMOTE = Integer.parseInt(param.getParameter(i++));
		if (param.getParameter(i).equalsIgnoreCase("both"))
			ASMO = 0;
		else if (param.getParameter(i).equalsIgnoreCase("minority"))
			ASMO = 1;
		else
			ASMO = 2;
		i++;
		if (param.getParameter(i++).equalsIgnoreCase("YES"))
			balance = true;
		else
			balance = false;
		smoting = Double.parseDouble(param.getParameter(i++));
		p = Double.parseDouble(param.getParameter(i++));
		nEvaluation = Integer.parseInt(param.getParameter(i++));
		cp = Double.parseDouble(param.getParameter(i++));
		bitWidth = Integer.parseInt(param.getParameter(i++));
	}
}