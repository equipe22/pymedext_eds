<script>
let response;
let result_json;
export let doc;
export let result;
export let txt = `
pas d'embolie pulmonaire.

antécédent d'infarctus chez le père.

poids: 67 kg, taille: 1m75, BMI: 45
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
  <textarea rows='20' bind:value={txt} on:input={handleAnnotate} /><br>
  <button on:click={handleAnnotate}>Annotate</button>
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
