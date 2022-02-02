<script>
let response;
let result_json;
export let doc;
export let result;
let entity_type = "regex"
export let txt = `
pas d'embolie pulmonaire.

présence d'EP. prof d'EPS à la retraite.

antécédent d'infarctus chez le père.

poids: 67 kg, taille: 1m75, BMI: 21,8
`;


export let regex = `
[  {
        "casesensitive": "yes",
        "libelle": "Embolie pulmonaire",
        "list_cui": "",
        "regexp": "[Ee]mbolie *[pP]ulmonaire|[^a-zA-Z]EP[^a-zA-Z]",
        "regexp_exclude": "",
        "id_regexp": "id_ep",
        "version": "v1"
    },
    {
        "index_extract": "3",
        "libelle": "Poids",
        "list_cui": "",
        "regexp": "(Poids|p[éeè]se)[^a-z0-9A-Z<>]*(de|est|est a|a)?[^a-z0-9A-Z<>,]*([0-9]+[.,]?[0-9]* *(kg|g)[0-9]*)",
        "regexp_exclude": "prise de poids",
        "id_regexp": "id_poids",
        "version": "v1"
    },
    {
        "index_extract": "4",
        "libelle": "IMC,BMI",
        "list_cui": "C1305855",
        "regexp": "(^|[^a-z])(IMC|BMI) *(est *de|de|=|:)? *([0-9]+[,.]?[0-9]*)[^\\/]?",
        "regexp_exclude": "",
        "id_regexp": "id_bmi",
        "version": "v1"
    }

]
`;

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

$: options= {
  'type': entity_type,
  'regex': JSON.parse(regex)
}

async function doPost () {
const res = await fetch('./annotate', {
  method: 'POST',
  body: JSON.stringify({'doc':[doc], 'options': options}),
  headers: {
  'Accept': 'application/json',
  'Content-Type': 'application/json'
}
})
response = await res.json()

result = response['html']


console.log(response['json'])

result_json = response['json']['annotations'].filter(obj => {
  return obj.type === options['type']
})
result_json = JSON.stringify(result_json, undefined, 2)

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
  margin: auto;
  width: 80%;
  text-align: left;
}

pre {
  margin: auto;
  width: 80%;
  padding-top: 2em;
  text-align: left;
}
</style>


<div>
  <textarea rows='20' bind:value={regex} /><br>
  <textarea rows='20' bind:value={txt}  /><br>
  <button on:click={handleAnnotate}>Annotate</button>
  <select bind:value={entity_type}>
    <option value="regex">regex</option>
    <option value="syntagme">syntagme</option>
    <option value="sentence">sentence</option>
  </select>
  {#await response}
  <p>annotating...</p>
  {:then}
  <h2>Results</h2>
  <p class='result'>{@html result}</p><br><br>
  <h2> JSON </h2>
  <pre> {result_json}</pre>
  {:catch error}
  <p style="color: red">{error.message}</p>
  {/await}
</div>
