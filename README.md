# Machine_learning_project
Project machine learning for Micheli exam

- struttura del input : [i,j] , i-esimo patter, j-esima feature
- struttura del layer : matrice [i,j] , i-esima unità del layer , j-esimo weight dell'output del layer precedente

TODO: In ordine

- Trovare metodo più furbo per fare validazione l'attributo di input viene cambiato due volte 
	a ogni step di apprendimento

- Adattare le funzioni a come le vuole Micheli (es. vedi se come errore vuole errore quadratico, 
						errore quadratico medio o altro)

- Implementare funzioni di splitting del dataset esternamente alla classe MLP e Layer

- Funzione di Rescaling/Standardizzazione dei dati (media 0, varianza 1 oppure range [0,1])

- Implementare funzione per salvare la rete (anche ad es. ogni N epoche (tramite flag booleana))

- Implementare minibatch

- Implementare momento

- Implementare nesterov

- Implementare funzione per creare diverse strategie di learning rate adattivo

- Implementare criteri di stopping

- Implementare Grid Search per scelta del modello


ROBE aggiuntive da fare solo se si rendono visibilmente necessarie:
- Cascade correlation
- Vari tipi di K fold cross validation
- Algortimi alternativi alla backpropagation: quickprop, RProp,...
- Dropout
