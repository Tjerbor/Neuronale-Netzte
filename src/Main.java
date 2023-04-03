
public class Main {


	public static void main( String [] args ) {
		
		NeuronalesNetz nn = new NeuronalesNetz();

		int[] struktur = {2,3,4,5};
 		nn.create( struktur );
 		// falls nicht anders vorgegeben:  
 		// - sollen alle Einheiten die Identitätsfunktion als Ausgabefunktion verwenden
 		// - sollen die Gewichte zufällig mit Werten zwschen -1 und 1 initialisiert werden
 		
 		System.out.println( nn ); 
 		// toString soll Netzkonfiguration (Ebenenstruktur und Gewichte) ausgeben
 		
		int[] strukturAND = {2,1}; // BIAS-Neuron intern verwaltet
 		nn.create( strukturAND );		
 		double [][][] w = nn.getWeights();
 		
 		nn.setUnitType(1, 0, "stepfun", 1.5);
 		
 		w[0][0][0] = 1.0;
 		w[0][1][0] = 1.0;
 		w[0][2][0] = 0.0; //BIAS-Neuron 'deaktivieren'
 		
 		nn.setWeights(w);
 		
 		System.out.println( nn );
 		
 		double[] x = { 1.0, 1.0 } ;
 		double[] yout = nn.compute( x );		
 		
 		double [][] data = {
 				{ 0.0, 0.0 },
 				{ 0.0, 1.0 },
 				{ 1.0, 0.0 },
 				{ 1.0, 1.0 }				
 		};
 		
 		double [][] out = nn.computeAll( data );		


	}
}
