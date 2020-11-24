<script>
let response;
let result_json = null;
export let doc;
export let result = null;
export let txt = `
Antécédents médicaux\n\n1,62 m pour 65,7 kg = IMC à 25,03.\nEn 1974 (19 ans), embolie pulmonaire compliquant une phlébite lors d\'un alitement à la suite d’un accident de la voie publique. Des bilans d\'hémostase effectués en 1995 et 2004 avaient éliminé une anomalie congénitale de la coagulation.\nInsuffisance veineuse saphène interne bilatérale connue depuis 6 ans pour laquelle avait été proposée une intervention.\nMode de vie\n\nAutonome et indépendante dans tous les gestes de la vie courante.\nLoisirs : course à pied, semi-marathon, randonnées dans le cadre d\'un club ainsi que pratique du vélo.\nLe chirurgien conclue :\n\n"Gonalgies droites évoluant depuis novembre 2012. Très sportive, a dû arrêter à cause de ses douleurs. Pas de blocage, ni d\'instabilité. Escaliers difficiles. Réveil nocturne. A eu plusieurs infiltrations d’acide hyaluronique sans aucun effet.\n\nExamen :\nBonnes amplitudes articulaires. Choc rotulien. Ligaments OK. Rotule OK. Douleurs palpation interne. Grinding test positif (NB : compression du genou en décubitus ventral en rotation interne qui, lorsqu\'il est douloureux, signale une atteinte méniscale). Pas de signe de tendinopathie.\n\nEcho et radios :\nSans particularité. IRM de 2012 normale.\n\nConduite à tenir :\nSymptomatologie méniscale. Refaire une IRM".\nLe 5 février 2014, 2e consultation avec le chirurgien orthopédiste :\n\n"Persistance des symptômes avec impotence fonctionnelle marquée. Réveils nocturnes...\n\nIRM :\nLésion de la CPMI (corne postérieure du ménisque interne) périphérique, aspect distendu du LCA (ligament croisé antérieur), kyste poplité.\n\nConduite à tenir :\nTélémétrie des membres inférieurs à la recherche de troubles d\'axes. Scintigraphie osseuse."\nLe 19 février 2014, 3e consultation avec le chirurgien orthopédiste :\n\n"Persistance des symptômes.\n\nScintigraphie osseuse :\nSouffrance sous chondrale du plateau tibial interne.\n\nConduite à tenir :\nPort de semelles valgisantes.\nEn fonction des douleurs, refaire une infiltration cortisonique."\nLe 7 novembre 2014, 4e consultation avec le chirurgien orthopédiste :\n\n"Persistance des douleurs, a eu des infiltrations de cortisone il y a une semaine.\n\nIRM :\nLésion instable de la corne postérieure du ménisque interne ouverte sur l\'articulation.\n\nConduite à tenir :\nIndication d\'arthroscopie pour méniscectomie interne, information sur la balance bénéfice/risque et les suites opératoires.\n\nPrescription :\nBilan sanguin comprenant numération formule sanguine, plaquettes, bilan de coagulation, ionogramme sanguin."\nConsentement éclairé, signé, daté du 07-11-2014 (exemplaire destiné à la clinique). L\'exemplaire destiné au médecin n\'était pas signé.\nLettre d\'information sur les risques liés à l\'intervention dans laquelle est précisée la possibilité d\'un traitement médicamenteux préventif contre la phlébite ainsi que la possibilité de phlébite en termes de complication.\nInformation sur la nécessité d\'une consultation pré anesthésie.\nLors de l’expertise, interrogée sur le contenu de ses entretiens avec le chirurgien orthopédiste, la patiente affirme l’avoir informé de ses antécédents de phlébite et d’embolie pulmonaire.\n\nPour sa part, le chirurgien orthopédiste reconnaît que, s\'il avait eu connaissance des antécédents de la patiente, il aurait mis en route un traitement préventif de la phlébite. En outre, il ajoute que s\'il avait eu cette information, celle-ci aurait été reportée sur ses notes de consultation.\n\nLe 6 janvier 2015, bilan biologique sans aucune anomalie, en particulier, avec des temps de coagulations normaux.\n\nLe 12 janvier 2015, consultation de préanesthésie. L’anesthésiste mentionne :\n\n"... Embolie pulmonaire à l\'âge de 19 ans suite à un accident de la voie publique..., Notion d’HTA. Risque ASA côté à 2".\nDans le questionnaire médical rempli par la patiente sont indiqués des antécédents familiaux d\'embolie pulmonaire et de phlébite, ainsi que des antécédents personnels de varice.\n\nLe 19 janvier 2015, hospitalisation en clinique (chirurgie ambulatoire).\n\nArthroscopie du genou droit : Méniscectomie latérale.\n\nCompte rendu opératoire : "... Au niveau du compartiment fémoro-tibial interne,\n\nménisque : lésion complexe instable de la corne moyenne et postérieure régularisation au shaver et à la pince baske.\ncartilage : lésion stade III …".\nL’anesthésiste ayant endormi la patiente n’était pas celui qui avait réalisé la consultation de préanesthésie.\n\nAux ordonnances destinées aux soins postopératoires, était jointe une note d\'information dans laquelle il était indiqué de contacter le chirurgien dans les cas suivants : "... Douleurs inhabituelles dans le mollet pouvant faire évoquer la survenue d\'une phlébite (caillots veineux dans la veine du mollet) …".\n\nLors de l’expertise, la patiente soulignait : "(…) s\'être étonnée et avoir réclamé au cours de son hospitalisation la mise en place d\'un traitement anticoagulant préventif. \nLe chirurgien lui aurait simplement indiqué qu\'elle pouvait marcher sans attelle et qu\'elle ne nécessitait pas d\'anticoagulant (…)".\n\nAprès sa sortie de la clinique, la patiente indiquait avoir marché avec une canne pendant 2 jours. Elle précisait avoir appelé le chirurgien dès la survenue de la douleur musculaire par un courriel du 21/01/15 à 11 h 38 (J2) : "Je ressens une douleur au mollet de la jambe opérée sans toutefois avoir la jambe chaude ou rouge, ni de température. Ayant eu une embolie pulmonaire il y a plus de 20 ans pourriez-vous me rassurer et échanger avec moi sur la conduite à tenir ?" \n\nA 12 h, le chirurgien rappelait la patiente et organisait un rendez-vous auprès d’un angiologue pour un echodoppler  ( N.B. : absence de notion de la prescription d’une HBPM à dose curative dès cet appel).\n\nLe 22 janvier 2015, consultation angiologue,\n\nEchodoppler :\n"Thrombose veineuse profonde distale suro-jambière droite récente évolutive. INNOHEP 0,6 fait, antivitamine K  à débuter  ce soir, contrôle dans 10 jours."\n\nOrdonnance :\n\nINNOHEP 0,6 ml x 1/j pendant 15 jours.\nPREVISCAN 3/4 de comprimé par jour le soir, posologie adaptée au dosage de l\'INR qui doit être compris entre 2 et 3.\n2 paires de bas jarret de compression classe 2.\nMarche.\nLe 24 février 2015 :\n\nEchodoppler : \n"Thrombose veineuse profonde distale suro-jambière droite récente évolutive sans extension poplité, très nettement reperméabilisée : disparition du thrombus soléaire, seul persiste le caillot péronier (9,5 mm)...". \nLe 27 février 2015, consultation du chirurgien :\n\n"amélioration des douleurs. A débuté la kiné avec retard. Pas de réveil nocturne. Pas d\'inflammation. Cicatrices OK. Sensibilité palpation interligne. Léger accrochage rotulien. Au total, bon résultat malgré les retards liés à la thrombose. Kiné ++. bilan dans 2 mois."\n\nOrdonnance de 10 séances de kinésithérapie du genou droit, trois fois par semaine au cabinet.\nAppui autorisé.\nLe 12 mars 2015 :\n\nEchodoppler : \n"Thrombose veineuse profonde péronière droite toujours obstructive au tiers moyen nécessitant la poursuite de l\'anticoagulation à dose curative associée à une compression élastique adaptée. Bilan d\'hémostase à prévoir à la recherche d’une probable anomalie constitutionnelle".\nLe 24 mars 2015 :\n\nEchodoppler : \n"Thrombose veineuse profonde péronière droite stable au tiers moyen."\nLe 20 avril 2015, service d’hématologie biologique du CHU :\n\n"Bilan pratiqué le 20 mars 2015 en raison d\'une embolie pulmonaire lors d\'une immobilisation en 1973 et d’une phlébite surale post-arthroscopie du genou droit en janvier 2015.\n\nLes prélèvements réalisés dans notre laboratoire en1995 et 2004, ainsi que le dernier du 20 mars 2015 sous PREVISCAN (INR à 1,99) ont permis d\'éliminer  toute anomalie de l\'hémostase  pour expliquer les épisodes thrombotiques présentés."\nLe 19 mai 2015 :\n\nEchodoppler :\n"Évolution favorable de la thrombose veineuse profonde jambière droite avec persistance d\'un petit syndrome obstructif. Anticoagulation à dose curative à poursuivre cet été et contrôle echodoppler à l\'automne. Poursuite de la compression élastique".\nLe 29 septembre 2015 :\n\nEchodoppler : \n"Évolution favorable de la thrombose veineuse profonde jambière droite avec recanalisation satisfaisante".\nDébut octobre 2015, arrêt du traitement par AVK .\n\nLors de l’expertise, la patiente indiquait : \n\n"avoir bénéficié, depuis mi-juin 2015, d\'un suivi psychiatrique, initialement 2 fois par semaine puis 1 fois par semaine, en raison d\'un état dépressif consécutif à la phlébite postopératoire. Celle-ci avait entraîné des réminiscences de sa précédente hospitalisation au décours de laquelle était survenue une phlébite avec embolie pulmonaire (en 1974). Elle avait été traitée par du PROZAC® en association avec de l\'ATARAX® qu\'elle prenait déjà…\n\nActuellement (juin 2016), elle était autonome et indépendante dans tous les gestes de la vie courante. Elle travaillait en mi-temps thérapeutique et désirait reprendre à temps complet.\n\nElle avait été améliorée par l\'intervention pratiquée par le chirurgien mais présentait des petites douleurs résiduelles sur la face interne du genou droit."
`;

$: console.log(JSON.stringify(result_json))

$: doc= {
  'annotations' : [{'type': 'raw_text',
                    'value': txt,
                    'span': [0,txt.length],
                    'ID': '2',
                    'source_ID': '1',
                    'source': 'raw'}],
//		'raw_text' : txt,
  'ID':'1',
  'source_ID': 'web'
}

async function doPost () {
const res = await fetch('./annotate', {
  method: 'POST',
  body: JSON.stringify([doc]),
  headers: {
  'Accept': 'application/json',
  'Content-Type': 'application/json'
}
})
response = await res.json()

result = response['html']
result_json = JSON.stringify(response['json'], undefined, 2)

}


function handleAnnotate() {
    doPost()
  }
</script>

<style>

textarea {
  height: auto;
  width: 80%;
}

.result {
  text-align: left;
}

pre {
  text-align: left;
}


</style>

<div>
  <textarea rows='20' bind:value={txt} /><br>
  <button on:click={handleAnnotate}>Annotate</button>
  <p class='result'>{@html result}</p>
  <pre> {result_json}</pre>
</div>
