<script>
let response;
let result_json;
export let doc;
export let result;
let entity_type = "regex"
export let txt = `pas d'embolie pulmonaire.
présence d'EP. prof d'EPS à la retraite.
antécédent d'infarctus chez le père.
Poids: 67 kg, taille: 1m75, BMI: 21,8
`;

export let regex = "(Poids|p[éeè]se)[^a-z0-9A-Z<>]*(de|est|est a|a)?[^a-z0-9A-Z<>,]*([0-9]+[.,]?[0-9]* *(kg|g)[0-9]*)"
export let regex_exclude = "prise de poids"
export let index_extract = "3";

$: doc= {
  'annotations' : [{'type': 'raw_text',
                    'value': txt,
                    'span': [0,txt.length],
                    'ID': '2',
                    'source_ID': '1',
                    'source': 'raw'}],
  'ID':'1',
  'source_ID': 'web'
}

$: options= {
  'type': entity_type,
  'regex': [{"casesensitive": "yes",
        "libelle": "my-regex",
        "index_extract": index_extract,
        "list_cui": "",
        "regexp": regex,
        "regexp_exclude": regex_exclude,
        "id_regexp": "id_regex",
        "version": "v0"}]
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


<div class='flex flex-col '>
  <div class='flex flex-row w-full p-4 bg-slate-100 border-2 rounded-lg border-slate-300  '>
    <div class='flex flex-col w-1/2 p-4'>
      <div class="mb-4 w-full flex-1">
        <label class="block text-gray-700 text-sm font-bold mb-2" for="regex">
          Regular expression
        </label>
        <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="regex" type="text" bind:value={regex}>

      </div>
      <div class="mb-4 w-full flex-1">
        <label class="block text-gray-700 text-sm font-bold mb-2" for="regex">
          Exclusion Regular expression
        </label>
        <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="exclusion" type="text" bind:value={regex_exclude}>
      </div>
      <div class="mb-4 w-full flex-1">
        <label class="block text-gray-700 text-sm font-bold mb-2" for="index_extract">
          Match group
        </label>
        <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="index_extract" type="text" bind:value={index_extract}>
      </div>
    </div>
    <div class='flex flex-col w-1/2 p-4'>
      <label for='examples' class="block text-gray-700 text-sm font-bold mb-2">Example sentences</label>
      <textarea id="examples" rows='7' bind:value={txt} class="w-full p-4" />
    </div>
  </div>
  <div class='flex flex-row mx-auto w-1/4 m-5'>
   
    <select bind:value={entity_type} class="form-select 
    block
    w-full
    px-3
    py-1.5
    text-base
    font-normal
    text-gray-700
    bg-white bg-clip-padding bg-no-repeat
    border border-solid border-gray-300
    rounded
    transition
    ease-in-out
    m-0
    focus:text-gray-700 focus:bg-white focus:border-blue-600 focus:outline-none mr-5">
      <option value="regex">regex</option>
      <option value="syntagme">syntagme</option>
      <option value="sentence">sentence</option>
    </select>
    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" on:click={handleAnnotate}>Annotate</button>
  </div>
 
  {#await response}
  <p>annotating...</p>
  {:then}
  <div class='flex flex-row mx-auto w-full'>
    <div class='flex flex-col mx-auto w-1/2 bg-stone-100 border-2 rounded-lg border-stone-300 mx-5 p-4'>
      <h2 class="block text-gray-700 text-sm font-bold mb-2">Results</h2>
      <p class='result text-left'>{@html result}</p><br><br>
    </div>
    <div class='flex flex-col mx-auto w-1/2 bg-stone-100 border-2 rounded-lg border-stone-300 mx-5 p-4'>
    <h2 class='block text-gray-700 text-sm font-bold mb-2'> JSON </h2>
    <pre class="text-left overflow-hide text-sm"> {result_json}</pre>
  </div>
  </div>
  {:catch error}
  <p style="color: red">{error.message}</p>
  {/await}
</div>
