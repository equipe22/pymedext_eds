<script>

//	import Random from './Random.svelte';
	import Annotate from './Annotate.svelte';



</script>

<main>
	<h1>PyMedExt_eds demo</h1>
	<h2>this demo uses the following pipeline : [endlines, sentences, family, syntagmes, negation, regex]</h2>
	<p>You can type your own text and click the annotate button to see the results</p>
	<Annotate />

</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

	h1 {
		color: #4287f5;
		font-size: 3em;
		font-weight: 100;
	}
	h2 {
		font-size: 1.5em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>
