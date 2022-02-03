<script>
//	import Random from './Random.svelte';
	import Annotate from './../lib/Annotate.svelte';

</script>

<main class='container text-center mx-auto'>

	<h1 class='text-5xl font-light my-5'>PyMedExt_eds demo</h1>
	<h2 class="text-xl mb-4">this demo uses the following pipeline : [PyRushSentences, family, syntagms, negation, regex]</h2>
	<p class="text-xl font-light mb-4">You can develop and test regular expressions interactively with a pymedext pipeline:</p>
	<Annotate />

</main>

<style>
	main {
	
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}


	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>
